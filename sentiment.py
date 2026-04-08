import aiohttp
import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable, Iterator
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys
import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ==========================
# 0. 基本配置（需要你修改的）
# ==========================

# Dify workflow endpoint aliases
HK_DIFY_API_URL = "http://192.168.10.97/v1/workflows/run"
SZ_DIFY_API_URL = "http://192.168.32.50/v1/workflows/run"

# Dify API keys / model profiles
GEMINI_DIFY_API_KEY = "app-mtVKZmpWyEOabRYkOR4WihPu"
GEMINI_BATCH_DIFY_API_KEY = "app-4sV7yFdvPuhiE0XSg4G4usAg"
GPT_DIFY_API_KEY = "app-SMU26wd8bd1VmfJTyhP2moe4"

# 根目录与数据表单目录配置
REPO_ROOT = Path(__file__).resolve().parent
DATA_FORM = "4"

# 数据目录：repo_root/data
DATA_DIR = REPO_ROOT / "data"

# 输入为 data_forms 级目录：repo_root/data/raw_data/data_form
DEFAULT_INPUT_DIR = DATA_DIR / "raw_data" / DATA_FORM

# Prompt 文件目录：repo_root/prompts/data_form.txt
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "4_v2.txt"

# 最大并发请求数（建议 5–20）
MAX_CONCURRENCY = 8 #DEFAULT 4 建议 8

# 每个 batch 的最大重试次数
MAX_RETRIES = 1

# 测试模式：只处理前 N 个待处理文件；None 或 <=0 表示不限制
TEST_FIRST_N_FILES = 50

# 输出目录：repo_root/data/results/data_form/prompt_name/
DEFAULT_OUTPUT_ROOT = DATA_DIR / "results" / DATA_FORM

DIFY_URL_ALIASES: Dict[str, str] = {
    "hk_api": HK_DIFY_API_URL,
    "sz_api": SZ_DIFY_API_URL,
}

DIFY_PROFILES: Dict[str, Dict[str, str]] = {
    "gemini": {
        "api_key": GEMINI_DIFY_API_KEY,
        "model_name": "gemini",
    },
    "gemi_batch": {
        "api_key": GEMINI_BATCH_DIFY_API_KEY,
        "model_name": "gemi_batch",
    },
    "gpt": {
        "api_key": GPT_DIFY_API_KEY,
        "model_name": "gpt",
    },
}

# ==========================
# 日志初始化（文件 + 控制台）
# ==========================

logger = logging.getLogger("sentiment_pipeline")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RuntimeConfig:
    dify_profile: str
    dify_url_alias: str
    dify_api_url: str
    dify_api_key: str
    model_name: str
    max_retries: int


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "model"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Form 4 sentiment analysis with configurable paths.")
    profile_help = ", ".join(
        f"{name} -> {cfg['model_name']}" for name, cfg in sorted(DIFY_PROFILES.items())
    )
    url_help = ", ".join(
        f"{name} -> {url}" for name, url in sorted(DIFY_URL_ALIASES.items())
    )
    parser.add_argument(
        "--dify-api",
        "--dify-profile",
        dest="dify_profile",
        choices=sorted(DIFY_PROFILES.keys()),
        default="gemini",
        help=f"Select which Dify app/profile to use. {profile_help}",
    )
    parser.add_argument(
        "--dify-url",
        choices=sorted(DIFY_URL_ALIASES.keys()),
        default="hk_api",
        help=f"Select which Dify workflow endpoint to use. {url_help}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing raw txt filings.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt text file used to build the Dify request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory for jsonl/parquet/log files. If omitted, defaults to data/results/4/<prompt_stem>_<model_name>.",
    )
    parser.add_argument(
        "--first-n-files",
        type=int,
        default=TEST_FIRST_N_FILES,
        help="Only process the first N pending files. Use 0 or a negative value for no limit.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="Maximum concurrent Dify requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Maximum retries per batch.",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Do not skip ids already present in the existing output files.",
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    dify_profile_cfg = DIFY_PROFILES[args.dify_profile]
    return RuntimeConfig(
        dify_profile=args.dify_profile,
        dify_url_alias=args.dify_url,
        dify_api_url=DIFY_URL_ALIASES[args.dify_url],
        dify_api_key=dify_profile_cfg["api_key"],
        model_name=sanitize_name(dify_profile_cfg["model_name"]),
        max_retries=int(args.max_retries),
    )


def build_runtime_paths(
    prompt_file_path: Path,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    result_dir_name = f"{prompt_file_path.stem}_{model_name}"
    results_base_dir = output_dir or (DEFAULT_OUTPUT_ROOT / result_dir_name)
    return {
        "results_base_dir": results_base_dir,
        "success_jsonl": results_base_dir / "sentiment_results.jsonl",
        "failed_jsonl": results_base_dir / "failed_transcripts.jsonl",
        "parquet": results_base_dir / "sentiment_results.parquet",
        "log_file": results_base_dir / "sentiment_pipeline.log",
    }


def setup_logger(log_file_path: Path) -> None:
    logger.handlers.clear()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


# ==========================
# 1. 批量情绪分析 Prompt 模板
# ==========================

DEFAULT_PROMPT_TEMPLATE = None
# DEFAULT_PROMPT_TEMPLATE = """
# You are a financial NLP analyst specializing in earnings call sentiment analysis.
# Your task is to analyze multiple earnings call transcripts in one request and output
# independent sentiment scores, confidence levels, and explanations for each transcript.

# -------------------------------------
# INPUT FORMAT
# -------------------------------------
# You will receive multiple transcripts in this structure:

# TRANSCRIPTS = [
#   {{"id": "T1", "text": "<< transcript content >>"}},
#   {{"id": "T2", "text": "<< transcript content >>"}},
#   ...
# ]

# Each transcript should be evaluated independently.
# Do NOT mix information between transcripts.

# Here are the transcripts to analyze:

# TRANSCRIPTS = __TRANSCRIPTS_JSON__

# -------------------------------------
# 1.SENTIMENT SCORE DEFINITION (VERY IMPORTANT)
# -------------------------------------
# Assign ONE overall sentiment score for each transcript.
# Score range: -1.0 to 1.0, increments of 0.1 only.

# Strongly Negative ( -1.0 to -0.6 ):
# - Significant misses in revenue/EPS/margins/cash flow
# - Management expresses clear caution, uncertainty, or deterioration
# - Lowered/withdrawn guidance or negative forward-looking comments
# - Defensive Q&A or recurring acknowledgment of severe challenges

# Slightly Negative ( -0.5 to -0.1 ):
# - Mild downward tone, small misses, or cautious commentary
# - Mixed performance but negatives outweigh positives
# - Guidance slightly softer than expected

# Neutral ( 0.0 ):
# - Results largely in line with expectations
# - Balanced tone with no clear directional bias

# Slightly Positive ( 0.1 to 0.5 ):
# - Mild beats or stable improvement
# - Constructive tone with growing confidence
# - Guidance slightly raised or encouraging commentary

# Strongly Positive ( 0.6 to 1.0 ):
# - Clear beats and strong execution
# - Confident and consistent messaging
# - Raised guidance or strong forward-looking indicators
# - Positive Q&A tone with strong demand or momentum signals

# -------------------------------------
# 2. **Generate a confidence score (0–100%)**
# -------------------------------------
#    Confidence should reflect:
#    - tone clarity  
#    - consistency of messaging  
#    - presence of quantifiable signals  
#    - transcript length and richness of cues  

# -------------------------------------
# 3. **Provide a structured explanation**:
# -------------------------------------
#    - Key positive factors  
#    - Key negative factors  
#    - Forward-looking guidance tone  
#    - Q&A tone and analyst reactions  


# -------------------------------------
# OUTPUT FORMAT (STRICT JSON ARRAY)
# -------------------------------------
# Output must be a JSON array with one object per transcript:

# [
#   {
#     "id": "T1",
#     "sentiment_score": <numeric value -1.0 to 1.0, step 0.1>,
#     "confidence": "<percentage, e.g., 84%>",
#     "summary": "<2–3 sentence summary of tone>",
#     "reasons": {
#         "positives": ["...", "..."],
#         "negatives": ["...", "..."],
#         "guidance_tone": "<...>",
#         "qa_tone": "<...>"
#     }
#   },
#   {
#     "id": "T2",
#     "sentiment_score": ...,
#     "confidence": "...",
#     "summary": "...",
#     "reasons": {...}
#   }
#   ...
# ]

# Rules:
# - The JSON must be valid and machine-readable.
# - Output MUST contain each transcript in the same order as the input.
# - Every transcript must get an independent score.
# - sentiment_score MUST adhere to the 0.1 increment rule.
# - Do NOT output anything outside the JSON array.
# """


def load_prompt_template(prompt_file_path: Path) -> str:
    """
    按 data_form 加载对应 prompt 文件。
    文件不存在时使用内置默认模板，避免流程中断。
    """
    if prompt_file_path.exists() and prompt_file_path.is_file():
        prompt_text = prompt_file_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            logger.warning(
                "Prompt file is empty, fallback to default template: %s",
                prompt_file_path,
            )
            return DEFAULT_PROMPT_TEMPLATE

        logger.info("Using prompt file: %s", prompt_file_path)
        return prompt_text

    logger.warning(
        "Prompt file not found, fallback to default template: %s",
        prompt_file_path,
    )
    return DEFAULT_PROMPT_TEMPLATE


# ==========================
# 2. 加载已有成功结果的 ID（用于断点续跑）
# ==========================

def load_processed_ids_from_jsonl(path: str) -> Set[str]:
    """
    从已存在的成功结果 JSONL 中读取已经处理过的 id 集合。
    如果文件不存在，返回空集合。
    """
    processed_ids: Set[str] = set()
    fp = Path(path)
    if not fp.exists():
        return processed_ids

    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("id")
                if tid:
                    processed_ids.add(tid)
            except json.JSONDecodeError:
                # 某一行坏掉就忽略，不影响整体
                continue

    logger.info(f"Loaded {len(processed_ids)} already-processed ids from {path}")
    return processed_ids


def load_processed_ids_from_parquet(path: str) -> Set[str]:
    processed_ids: Set[str] = set()
    fp = Path(path)
    if not fp.exists():
        return processed_ids

    try:
        df = pd.read_parquet(fp, columns=["id"])
    except Exception as exc:
        logger.warning("Failed to read existing parquet ids from %s: %s", path, exc)
        return processed_ids

    if "id" not in df.columns:
        return processed_ids

    processed_ids = {str(value) for value in df["id"].dropna().tolist()}
    logger.info("Loaded %s already-processed ids from %s", len(processed_ids), path)
    return processed_ids


def load_processed_ids(success_jsonl_path: str, parquet_path: str) -> Set[str]:
    processed_ids = load_processed_ids_from_jsonl(success_jsonl_path)
    processed_ids.update(load_processed_ids_from_parquet(parquet_path))
    return processed_ids


# ==========================
# 3. 统计 + 流式读取 txt，过滤已处理的
# ==========================

def count_pending_transcripts(
    dir_path: str,
    skip_ids: Set[str],
    first_n_files: Optional[int] = None
) -> int:
    """
    统计 data_form 级目录下待处理 transcript 数量（仅计数，不读取文件内容）。
    id = 文件名去掉扩展名。
    已在 skip_ids 里的 id 会被跳过（断点续跑用）。
    first_n_files 仅在测试模式下使用：限制最终数量。
    """
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        logger.error(f"Directory not found or not a directory: {dir_path}")
        return 0

    symbol_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not symbol_dirs:
        logger.warning(f"No symbol directories found under: {dir_path}")

    total_txt_files = 0
    pending_count = 0
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None

    for symbol_dir in symbol_dirs:
        id_dirs = sorted([p for p in symbol_dir.iterdir() if p.is_dir()])
        for id_dir in id_dirs:
            txt_files = sorted(id_dir.glob("*.txt"))
            for fp in txt_files:
                total_txt_files += 1
                tid = fp.stem
                if tid in skip_ids:
                    continue

                pending_count += 1
                if effective_limit is not None and pending_count >= effective_limit:
                    logger.info(
                        "[TEST MODE] first_n_files=%s, pending=%s, reached_limit=True",
                        effective_limit,
                        pending_count,
                    )
                    return pending_count

    if total_txt_files == 0:
        logger.warning(f"No .txt files found under symbol/id folders in: {dir_path}")

    logger.info("Pending transcripts to process: %s", pending_count)
    if effective_limit is not None:
        logger.info(
            "[TEST MODE] first_n_files=%s, pending=%s, reached_limit=False",
            effective_limit,
            pending_count,
        )
    return pending_count


def iter_transcripts_from_data_form_dir(
    dir_path: str,
    skip_ids: Set[str],
    first_n_files: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    从 data_form 级目录递归流式读取：symbol/id/*.txt。
    按需 yield transcript，避免一次性把全部文本载入内存。
    """
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        logger.error(f"Directory not found or not a directory: {dir_path}")
        return

    symbol_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not symbol_dirs:
        logger.warning(f"No symbol directories found under: {dir_path}")
        return

    selected_count = 0
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None

    for symbol_dir in symbol_dirs:
        id_dirs = sorted([p for p in symbol_dir.iterdir() if p.is_dir()])
        for id_dir in id_dirs:
            txt_files = sorted(id_dir.glob("*.txt"))
            for fp in txt_files:
                tid = fp.stem
                if tid in skip_ids:
                    continue

                if effective_limit is not None and selected_count >= effective_limit:
                    return

                try:
                    text = fp.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read {fp} with utf-8, trying latin-1")
                    text = fp.read_text(encoding="latin-1")

                selected_count += 1
                yield {
                    "id": tid,
                    "text": text,
                    "symbol": symbol_dir.name,
                    "source_file": str(fp),
                }


# ==========================
# 4. 分批（每批最多 5 个）
# ==========================

def chunk_transcripts(
    transcripts: Iterable[Dict[str, Any]],
    batch_size: int = 5
) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in transcripts:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# ==========================
# 5. 把成功结果增量写入 JSONL（每条结果一行）
# ==========================

def append_success_results_to_jsonl(
    batch_result: List[Dict[str, Any]],
    path: str
) -> None:
    """
    将一个 batch 的成功结果追加写入 success JSONL。
    每个 item 必须包含 'id' 字段。
    """
    if not batch_result:
        return
    with open(path, "a", encoding="utf-8") as f:
        for item in batch_result:
            # 确保有 id
            if "id" not in item:
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_failed_records_to_jsonl(
    failed_records: List[Dict[str, Any]],
    path: str
) -> None:
    """
    将失败批次记录追加写入失败 JSONL。
    """
    if not failed_records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json_with_auto_escape(s: str, max_fixes: int = 50):
    """
    自动修复 JSON 文本中字符串里的未转义双引号，然后 json.loads。
    只适合：
      - 整体结构本来就是 JSON
      - 偶尔有 "xxx" 这种没转义的内嵌引号
    """
    text = s
    for _ in range(max_fixes):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # 只针对类似 “Expecting ',' delimiter” 这类情况做修复
            msg = str(e)
            if "Expecting ',' delimiter" not in msg and "Expecting ':' delimiter" not in msg:
                # 其他类型的错误先不要瞎修
                raise

            error_pos = e.pos  # 出错的大致位置
            i = error_pos - 1

            # 从报错位置往前找最近一个双引号
            while i >= 0 and text[i] != '"':
                i -= 1

            if i < 0:
                # 实在找不到就放弃
                raise

            # 把这个双引号替换成转义版本 \"
            text = text[:i] + '\\"' + text[i+1:]
    # 修了很多次还不行，就认为不是这种简单问题
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise

# ==========================
# 6. 调用 Dify 的异步函数（带重试 + 报错处理）
# ==========================

async def call_dify_batch(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    batch: List[Dict[str, Any]],
    failed_records: List[Dict[str, Any]],
    prompt_template: str,
    runtime_config: RuntimeConfig,
) -> Optional[List[Dict[str, Any]]]:
    transcripts_payload = [
        {"id": t["id"], "text": t["text"]}
        for t in batch
    ]
    transcripts_json_str = json.dumps(transcripts_payload, ensure_ascii=False)

    full_prompt = prompt_template.replace("__TRANSCRIPTS_JSON__", transcripts_json_str)


    payload = {
        "inputs": {
            "input": full_prompt
        },
        "response_mode": "blocking",
        "user": f"earnings-sentiment-batch-{runtime_config.model_name}"
    }

    headers = {
        "Authorization": f"Bearer {runtime_config.dify_api_key}",
        "Content-Type": "application/json",
    }

    last_error_reason = None

    for attempt in range(1, runtime_config.max_retries + 1):
        try:
            async with sem:
                async with session.post(
                    runtime_config.dify_api_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as resp:
                    status = resp.status
                    text = await resp.text()

            if status != 200:
                last_error_reason = f"HTTP {status}: {text[:200]}"
                logger.error(f"HTTP {status} on attempt {attempt}: {text[:200]}")

                if 400 <= status < 500 and status != 429:
                    break

            else:
                try:
                    data = json.loads(text)['data']
                except json.JSONDecodeError:
                    last_error_reason = "Response is not valid JSON"
                    logger.error(f"Response is not valid JSON on attempt {attempt}")
                    await asyncio.sleep(2 * attempt)
                    continue

                # 根据你的 Dify 返回结构调整这里
                answer = data.get("outputs").get("text")
                if not answer:
                    last_error_reason = "No 'answer' field in response"
                    logger.error(f"No 'answer' field in response: {data}")
                    await asyncio.sleep(2 * attempt)
                    continue

                answer_str = answer.strip()
                try:
                    # 只解析第一个 JSON 值（就是你要的那个数组）
                    match = re.search(r"\[\s*\{.*?\}\s*\]", answer_str, re.S)
                    json_text = match.group(0)
                    # result_json_array = json.loads(json_text)
                    result_json_array = load_json_with_auto_escape(json_text)
                except json.JSONDecodeError:
                    last_error_reason = "'answer' field is not valid JSON (even after 're.search')"
                    logger.error(f"'answer' field is not valid JSON: {answer_str[:200]}")
                    await asyncio.sleep(2 * attempt)
                    continue

                if not isinstance(result_json_array, list):
                    last_error_reason = "Parsed answer is not a list"
                    logger.error(f"Parsed answer is not a list: {result_json_array}")
                    await asyncio.sleep(2 * attempt)
                    continue

                if len(result_json_array) != len(batch):
                    logger.warning(
                        f"Result length {len(result_json_array)} != batch length {len(batch)}"
                    )

                return result_json_array

        except asyncio.TimeoutError:
            last_error_reason = "Timeout"
            logger.error(f"Timeout on attempt {attempt}")
        except aiohttp.ClientError as e:
            last_error_reason = f"Network error: {e}"
            logger.error(f"Network error on attempt {attempt}: {e}")

        await asyncio.sleep(2 * attempt)

    batch_ids = [t["id"] for t in batch]
    logger.error(f"Batch with ids {batch_ids} failed after {runtime_config.max_retries} attempts.")

    failed_records.append({
        "batch_ids": batch_ids,
        "reason": last_error_reason or "Unknown error",
        "attempts": runtime_config.max_retries,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

    return None


# ==========================
# 7. 主异步函数：并发跑所有 transcripts（同时增量写 JSONL）
# ==========================

async def analyze_transcripts_concurrently(
    transcripts: Iterable[Dict[str, Any]],
    prompt_template: str,
    total_to_process: int,
    success_jsonl_path: str,
    runtime_config: RuntimeConfig,
    max_concurrency: int = MAX_CONCURRENCY
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    流式消费 transcripts 并有界并发调用 Dify，避免全量文本常驻内存。
    返回本轮统计信息；持久化结果在过程中已写入 JSONL。
    """
    failed_records_all: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_concurrency)

    processed_count = 0
    success_count = 0
    failed_count = 0

    logger.info(f"Start processing {total_to_process} transcripts in this run.")

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_to_process,
            desc="Sentiment batches",
            unit="transcript",
            dynamic_ncols=True,
        )

    async with aiohttp.ClientSession() as session:
        pending_tasks: Set[asyncio.Task] = set()
        task_meta: Dict[asyncio.Task, Dict[str, Any]] = {}

        async def process_done(done_tasks: Set[asyncio.Task]) -> None:
            nonlocal processed_count, success_count, failed_count

            for done_task in done_tasks:
                meta = task_meta.pop(done_task, {"size": 0, "ids": []})
                batch_size = meta.get("size", 0)
                batch_ids = meta.get("ids", [])

                try:
                    batch_result = done_task.result()
                except Exception as e:
                    batch_result = None
                    failed_records_all.append({
                        "batch_ids": batch_ids,
                        "reason": f"Unhandled task exception: {e}",
                        "attempts": runtime_config.max_retries,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })

                processed_count += batch_size
                remaining = total_to_process - processed_count
                progress = processed_count / total_to_process if total_to_process > 0 else 1.0

                logger.info(
                    f"[PROGRESS] Processed {processed_count}/{total_to_process} transcripts "
                    f"({progress:.2%}), remaining {remaining}"
                )

                if pbar is not None:
                    pbar.update(batch_size)

                if batch_result is None:
                    failed_count += batch_size
                    if pbar is not None:
                        pbar.set_postfix(success=success_count, failed=failed_count)
                    continue

                append_success_results_to_jsonl(batch_result, success_jsonl_path)

                success_count += len(batch_result)
                failed_count += max(0, batch_size - len(batch_result))
                if pbar is not None:
                    pbar.set_postfix(success=success_count, failed=failed_count)

        for batch in chunk_transcripts(transcripts, batch_size=5):
            task = asyncio.create_task(
                call_dify_batch(
                    session,
                    sem,
                    batch,
                    failed_records_all,
                    prompt_template,
                    runtime_config,
                )
            )
            pending_tasks.add(task)
            task_meta[task] = {
                "size": len(batch),
                "ids": [t["id"] for t in batch],
            }

            if len(pending_tasks) >= max_concurrency:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await process_done(done)

        while pending_tasks:
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            await process_done(done)

    if pbar is not None:
        pbar.close()

    logger.info("All batches for this run have completed.")
    stats = {
        "processed": processed_count,
        "success": success_count,
        "failed": failed_count,
        "total": total_to_process,
    }
    return stats, failed_records_all


# ==========================
# 8. 从成功 JSONL 转为 Parquet
# ==========================

def convert_success_jsonl_to_parquet(
    success_jsonl_path: str,
    parquet_path: str
) -> None:
    """
    读取所有成功结果 JSONL，展平为表结构，写入 Parquet。
    可重复执行，每次都会完全根据 JSONL 重建 Parquet。
    """
    def build_row(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tid = obj.get("id")
        if not tid:
            return None

        reasons = obj.get("reasons", {}) or {}
        positives = reasons.get("positives", [])
        negatives = reasons.get("negatives", [])
        guidance_tone = reasons.get("guidance_tone", "")
        qa_tone = reasons.get("qa_tone", "")

        return {
            "id": str(tid),
            "sentiment_score": obj.get("sentiment_score"),
            "confidence": obj.get("confidence"),
            "summary": obj.get("summary"),
            "reasons_positives": json.dumps(positives, ensure_ascii=False),
            "reasons_negatives": json.dumps(negatives, ensure_ascii=False),
            "reasons_guidance_tone": guidance_tone,
            "reasons_qa_tone": qa_tone,
        }

    rows: List[Dict[str, Any]] = []
    parquet_fp = Path(parquet_path)
    jsonl_fp = Path(success_jsonl_path)

    if parquet_fp.exists():
        try:
            existing_df = pd.read_parquet(parquet_fp)
            if not existing_df.empty and "id" in existing_df.columns:
                rows.extend(existing_df.to_dict(orient="records"))
        except Exception as exc:
            logger.warning("Failed to read existing parquet %s: %s", parquet_path, exc)

    if jsonl_fp.exists():
        with jsonl_fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row = build_row(obj)
                if row is not None:
                    rows.append(row)

    if not rows:
        print(f"[INFO] No valid rows found in {success_jsonl_path} or {parquet_path}, skip Parquet.")
        return

    df = pd.DataFrame(rows)
    if "id" not in df.columns:
        print("[INFO] No id column found after merge, skip Parquet.")
        return

    df["id"] = df["id"].astype(str)
    merged_count = len(df)
    df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    parquet_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_fp, index=False)
    print(
        f"[INFO] Merged {merged_count} rows into {len(df)} unique ids and wrote Parquet: {parquet_path}"
    )


# ==========================
# 9. 入口：断点续跑 + 转 Parquet
# ==========================

if __name__ == "__main__":
    args = build_parser().parse_args()
    runtime_config = build_runtime_config(args)
    prompt_file_path = args.prompt_file
    input_dir = str(args.input_dir)
    first_n_files = args.first_n_files
    if first_n_files is not None and first_n_files <= 0:
        first_n_files = None

    runtime_paths = build_runtime_paths(prompt_file_path, runtime_config.model_name, args.output_dir)
    setup_logger(runtime_paths["log_file"])
    prompt_template = load_prompt_template(prompt_file_path)

    logger.info("Using Dify profile: %s", runtime_config.dify_profile)
    logger.info("Using Dify URL alias: %s", runtime_config.dify_url_alias)
    logger.info("Resolved Dify URL: %s", runtime_config.dify_api_url)
    logger.info("Resolved model name: %s", runtime_config.model_name)
    logger.info("Using input dir: %s", input_dir)
    logger.info("Using prompt file: %s", prompt_file_path)
    logger.info("Using output dir: %s", runtime_paths["results_base_dir"])

    # 1）先看之前已经成功的有哪些，跳过
    processed_ids: Set[str] = set()
    if not args.reprocess_existing:
        processed_ids = load_processed_ids(
            str(runtime_paths["success_jsonl"]),
            str(runtime_paths["parquet"]),
        )

    # 2）先统计本轮要处理的数量（不读文件正文）
    pending_total = count_pending_transcripts(
        input_dir,
        processed_ids,
        first_n_files=first_n_files,
    )

    if pending_total <= 0:
        print("[INFO] No new transcripts to process. You can still regenerate Parquet from existing JSONL.")
        # 直接从已有 JSONL 生成一次 Parquet
        convert_success_jsonl_to_parquet(
            str(runtime_paths["success_jsonl"]),
            str(runtime_paths["parquet"]),
        )
        sys.exit(0)

    # 3）流式加载 transcripts（边读边处理）
    transcripts_iter = iter_transcripts_from_data_form_dir(
        input_dir,
        processed_ids,
        first_n_files=first_n_files,
    )

    # 4）并发分析 + 增量写 JSONL
    stats, failed_records = asyncio.run(
        analyze_transcripts_concurrently(
            transcripts_iter,
            prompt_template,
            total_to_process=pending_total,
            success_jsonl_path=str(runtime_paths["success_jsonl"]),
            runtime_config=runtime_config,
            max_concurrency=int(args.max_concurrency),
        )
    )

    failed_ids = [tid for rec in failed_records for tid in rec.get("batch_ids", [])]

    print(f"[SUMMARY] Dify profile: {runtime_config.dify_profile}")
    print(f"[SUMMARY] Dify URL   : {runtime_config.dify_url_alias} -> {runtime_config.dify_api_url}")
    print(f"[SUMMARY] Model name : {runtime_config.model_name}")
    print(f"[SUMMARY] Output dir : {runtime_paths['results_base_dir']}")
    print(f"[SUMMARY] Newly processed transcripts: {stats['processed']}")
    print(f"[SUMMARY] Success count (this run): {stats['success']}")
    print(f"[SUMMARY] Failed count  (this run): {stats['failed']}")
    if failed_ids:
        print(f"[SUMMARY] Failed transcript ids (this run): {failed_ids}")

    # 5）追加写失败批次日志
    append_failed_records_to_jsonl(failed_records, str(runtime_paths["failed_jsonl"]))

    # 6）基于完整的 success JSONL 重新生成一次 Parquet
    convert_success_jsonl_to_parquet(
        str(runtime_paths["success_jsonl"]),
        str(runtime_paths["parquet"]),
    )
