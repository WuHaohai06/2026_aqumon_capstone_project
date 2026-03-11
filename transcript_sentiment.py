import aiohttp
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys
import logging
import time


# ==========================
# 0. 基本配置（需要你修改的）
# ==========================

# 替换为你自己的 Dify 应用接口地址
DIFY_API_URL = "http://192.168.10.97/v1/workflows/run"

# 替换为你自己的 Dify API Key
# gemini2.5 flash for capstone
DIFY_API_KEY = "app-mtVKZmpWyEOabRYkOR4WihPu"

# 根目录与数据表单目录配置
ROOT_DIR = "./data"
DATA_FORM = "4"

# 输入为 data_forms 级目录：root_dir/raw_data/data_forms
DATA_FORM_INPUT_DIR = str(Path(ROOT_DIR) / "raw_data" / DATA_FORM)

# 最大并发请求数（建议 5–20）
MAX_CONCURRENCY = 10

# 每个 batch 的最大重试次数
MAX_RETRIES = 1

# 控制台进度条宽度
PROGRESS_BAR_WIDTH = 30

# 输出目录：root_dir/results/data_forms/
RESULTS_BASE_DIR = Path(ROOT_DIR) / "results" / DATA_FORM
SUCCESS_RESULTS_JSONL_PATH = str(RESULTS_BASE_DIR / "sentiment_results.jsonl")
FAILED_LOG_JSONL_PATH = str(RESULTS_BASE_DIR / "failed_transcripts.jsonl")
OUTPUT_PARQUET_PATH = str(RESULTS_BASE_DIR / "earnings_sentiment.parquet")
LOG_FILE_PATH = str(RESULTS_BASE_DIR / "sentiment_pipeline.log")

# ==========================
# 日志初始化（文件 + 控制台）
# ==========================

logger = logging.getLogger("sentiment_pipeline")
logger.setLevel(logging.INFO)

# 确保日志与结果目录存在
RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# 防止重复添加 handler
if not logger.handlers:
    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 文件
    fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


# ==========================
# 1. 批量情绪分析 Prompt 模板
# ==========================

PROMPT_TEMPLATE = """
You are a financial NLP analyst specializing in earnings call sentiment analysis.
Your task is to analyze multiple earnings call transcripts in one request and output
independent sentiment scores, confidence levels, and explanations for each transcript.

-------------------------------------
INPUT FORMAT
-------------------------------------
You will receive multiple transcripts in this structure:

TRANSCRIPTS = [
  {{"id": "T1", "text": "<< transcript content >>"}},
  {{"id": "T2", "text": "<< transcript content >>"}},
  ...
]

Each transcript should be evaluated independently.
Do NOT mix information between transcripts.

Here are the transcripts to analyze:

TRANSCRIPTS = __TRANSCRIPTS_JSON__

-------------------------------------
1.SENTIMENT SCORE DEFINITION (VERY IMPORTANT)
-------------------------------------
Assign ONE overall sentiment score for each transcript.
Score range: -1.0 to 1.0, increments of 0.1 only.

Strongly Negative ( -1.0 to -0.6 ):
- Significant misses in revenue/EPS/margins/cash flow
- Management expresses clear caution, uncertainty, or deterioration
- Lowered/withdrawn guidance or negative forward-looking comments
- Defensive Q&A or recurring acknowledgment of severe challenges

Slightly Negative ( -0.5 to -0.1 ):
- Mild downward tone, small misses, or cautious commentary
- Mixed performance but negatives outweigh positives
- Guidance slightly softer than expected

Neutral ( 0.0 ):
- Results largely in line with expectations
- Balanced tone with no clear directional bias

Slightly Positive ( 0.1 to 0.5 ):
- Mild beats or stable improvement
- Constructive tone with growing confidence
- Guidance slightly raised or encouraging commentary

Strongly Positive ( 0.6 to 1.0 ):
- Clear beats and strong execution
- Confident and consistent messaging
- Raised guidance or strong forward-looking indicators
- Positive Q&A tone with strong demand or momentum signals

-------------------------------------
2. **Generate a confidence score (0–100%)**
-------------------------------------
   Confidence should reflect:
   - tone clarity  
   - consistency of messaging  
   - presence of quantifiable signals  
   - transcript length and richness of cues  

-------------------------------------
3. **Provide a structured explanation**:
-------------------------------------
   - Key positive factors  
   - Key negative factors  
   - Forward-looking guidance tone  
   - Q&A tone and analyst reactions  


-------------------------------------
OUTPUT FORMAT (STRICT JSON ARRAY)
-------------------------------------
Output must be a JSON array with one object per transcript:

[
  {
    "id": "T1",
    "sentiment_score": <numeric value -1.0 to 1.0, step 0.1>,
    "confidence": "<percentage, e.g., 84%>",
    "summary": "<2–3 sentence summary of tone>",
    "reasons": {
        "positives": ["...", "..."],
        "negatives": ["...", "..."],
        "guidance_tone": "<...>",
        "qa_tone": "<...>"
    }
  },
  {
    "id": "T2",
    "sentiment_score": ...,
    "confidence": "...",
    "summary": "...",
    "reasons": {...}
  }
  ...
]

Rules:
- The JSON must be valid and machine-readable.
- Output MUST contain each transcript in the same order as the input.
- Every transcript must get an independent score.
- sentiment_score MUST adhere to the 0.1 increment rule.
- Do NOT output anything outside the JSON array.
"""


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


# ==========================
# 3. 从目录读取 txt，过滤已处理的
# ==========================

def load_transcripts_from_data_form_dir(dir_path: str, skip_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    从 data_form 级目录递归读取：symbol/id/*.txt。
    id = 文件名去掉扩展名。
    已在 skip_ids 里的 id 会被跳过（断点续跑用）。
    """
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        logger.error(f"Directory not found or not a directory: {dir_path}")
        return []

    transcripts: List[Dict[str, Any]] = []

    symbol_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not symbol_dirs:
        logger.warning(f"No symbol directories found under: {dir_path}")

    total_txt_files = 0
    total_symbol_dirs = len(symbol_dirs)
    total_id_dirs = 0
    skipped_count = 0
    for symbol_dir in symbol_dirs:
        id_dirs = sorted([p for p in symbol_dir.iterdir() if p.is_dir()])
        total_id_dirs += len(id_dirs)
        for id_dir in id_dirs:
            txt_files = sorted(id_dir.glob("*.txt"))
            for fp in txt_files:
                total_txt_files += 1
                tid = fp.stem
                if tid in skip_ids:
                    skipped_count += 1
                    continue

                try:
                    text = fp.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read {fp} with utf-8, trying latin-1")
                    text = fp.read_text(encoding="latin-1")

                transcripts.append({
                    "id": tid,
                    "text": text,
                    "symbol": symbol_dir.name,
                    "source_file": str(fp),
                })

    if total_txt_files == 0:
        logger.warning(f"No .txt files found under symbol/id folders in: {dir_path}")

    logger.info(
        f"Loaded {len(transcripts)} transcripts from data_form directory {dir_path} "
        f"(after skipping processed)"
    )
    logger.info(
        "[SCAN] symbols=%s, id_dirs=%s, txt_files=%s, skipped=%s, to_process=%s",
        total_symbol_dirs,
        total_id_dirs,
        total_txt_files,
        skipped_count,
        len(transcripts),
    )
    return transcripts


# ==========================
# 4. 分批（每批最多 5 个）
# ==========================

def chunk_transcripts(transcripts: List[Dict[str, Any]], batch_size: int = 5):
    for i in range(0, len(transcripts), batch_size):
        yield transcripts[i:i + batch_size]


def format_progress_bar(progress: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    """
    将 0-1 进度值格式化为文本进度条。
    """
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


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
    failed_records: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    transcripts_payload = [
        {"id": t["id"], "text": t["text"]}
        for t in batch
    ]
    transcripts_json_str = json.dumps(transcripts_payload, ensure_ascii=False)

    full_prompt = PROMPT_TEMPLATE.replace("__TRANSCRIPTS_JSON__", transcripts_json_str)


    payload = {
        "inputs": {
            "input": full_prompt
        },
        "response_mode": "blocking",
        "user": "earnings-sentiment-batch"
    }

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error_reason = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                async with session.post(
                    DIFY_API_URL,
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
    logger.error(f"Batch with ids {batch_ids} failed after {MAX_RETRIES} attempts.")

    failed_records.append({
        "batch_ids": batch_ids,
        "reason": last_error_reason or "Unknown error",
        "attempts": MAX_RETRIES,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

    return None


# ==========================
# 7. 主异步函数：并发跑所有 transcripts（同时增量写 JSONL）
# ==========================

async def analyze_transcripts_concurrently(
    transcripts: List[Dict[str, Any]],
    max_concurrency: int = MAX_CONCURRENCY
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    返回 results_by_id 只是为了给你一个汇总视图；
    真正的持久化结果已经在过程中写进 JSONL 了。
    """
    results_by_id: Dict[str, Any] = {}
    failed_records_all: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_concurrency)

    total_to_process = len(transcripts)
    processed_count = 0
    success_count = 0
    failed_count = 0
    total_batches = (total_to_process + 4) // 5
    start_ts = time.perf_counter()

    logger.info(f"Start processing {total_to_process} transcripts in this run.")
    logger.info(
        "[DEBUG] max_concurrency=%s, batch_size=%s, total_batches=%s",
        max_concurrency,
        5,
        total_batches,
    )

    async with aiohttp.ClientSession() as session:
        tasks = []
        for batch in chunk_transcripts(transcripts, batch_size=5):
            task = asyncio.create_task(
                call_dify_batch(session, sem, batch, failed_records_all)
            )
            tasks.append((batch, task))

        for batch_index, (batch, task) in enumerate(tasks, start=1):
            batch_result = await task

            processed_count += len(batch)
            remaining = total_to_process - processed_count
            progress = processed_count / total_to_process if total_to_process > 0 else 1.0
            elapsed = time.perf_counter() - start_ts
            speed = processed_count / elapsed if elapsed > 0 else 0.0
            eta_seconds = (remaining / speed) if speed > 0 else 0.0
            bar = format_progress_bar(progress)

            if batch_result is None:
                failed_count += len(batch)
                logger.info(
                    "[PROGRESS] batch=%s/%s %s %.2f%% | processed=%s/%s | success=%s failed=%s | "
                    "speed=%.2f tx/s | eta=%.1fs | batch_status=FAILED",
                    batch_index,
                    total_batches,
                    bar,
                    progress * 100,
                    processed_count,
                    total_to_process,
                    success_count,
                    failed_count,
                    speed,
                    eta_seconds,
                )
                for t in batch:
                    results_by_id[t["id"]] = None
                continue

            # 写入成功结果 JSONL（增量）
            append_success_results_to_jsonl(batch_result, SUCCESS_RESULTS_JSONL_PATH)
            success_count += len(batch_result)
            failed_count += max(0, len(batch) - len(batch_result))

            logger.info(
                "[PROGRESS] batch=%s/%s %s %.2f%% | processed=%s/%s | success=%s failed=%s | "
                "speed=%.2f tx/s | eta=%.1fs | batch_status=OK",
                batch_index,
                total_batches,
                bar,
                progress * 100,
                processed_count,
                total_to_process,
                success_count,
                failed_count,
                speed,
                eta_seconds,
            )

            # 同时在内存里也留一份
            for item in batch_result:
                tid = item.get("id")
                if tid is None:
                    logger.warning(f"One result item has no 'id': {item}")
                    continue
                results_by_id[tid] = item

    total_elapsed = time.perf_counter() - start_ts
    logger.info(
        "All batches completed. elapsed=%.2fs, processed=%s, success=%s, failed=%s",
        total_elapsed,
        total_to_process,
        success_count,
        failed_count,
    )
    return results_by_id, failed_records_all


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
    fp = Path(success_jsonl_path)
    if not fp.exists():
        print(f"[INFO] No success JSONL found at {success_jsonl_path}, skip Parquet conversion.")
        return

    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            tid = obj.get("id")
            reasons = obj.get("reasons", {}) or {}
            positives = reasons.get("positives", [])
            negatives = reasons.get("negatives", [])
            guidance_tone = reasons.get("guidance_tone", "")
            qa_tone = reasons.get("qa_tone", "")

            row = {
                "id": tid,
                "sentiment_score": obj.get("sentiment_score"),
                "confidence": obj.get("confidence"),
                "summary": obj.get("summary"),
                "reasons_positives": json.dumps(positives, ensure_ascii=False),
                "reasons_negatives": json.dumps(negatives, ensure_ascii=False),
                "reasons_guidance_tone": guidance_tone,
                "reasons_qa_tone": qa_tone,
            }
            rows.append(row)

    if not rows:
        print("[INFO] No valid rows parsed from success JSONL, skip Parquet.")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Converted {len(rows)} rows from JSONL to Parquet: {parquet_path}")


# ==========================
# 9. 入口：断点续跑 + 转 Parquet
# ==========================

if __name__ == "__main__":
    # 1）先看之前已经成功的有哪些，跳过
    processed_ids = load_processed_ids_from_jsonl(SUCCESS_RESULTS_JSONL_PATH)

    # 2）加载还没处理过的 transcripts
    transcripts = load_transcripts_from_data_form_dir(DATA_FORM_INPUT_DIR, processed_ids)
    # transcripts = None

    if not transcripts:
        print("[INFO] No new transcripts to process. You can still regenerate Parquet from existing JSONL.")
        # 直接从已有 JSONL 生成一次 Parquet
        convert_success_jsonl_to_parquet(SUCCESS_RESULTS_JSONL_PATH, OUTPUT_PARQUET_PATH)
        sys.exit(0)

    # 3）并发分析 + 增量写 JSONL
    results_by_id, failed_records = asyncio.run(
        analyze_transcripts_concurrently(transcripts)
    )

    success_ids = [tid for tid, res in results_by_id.items() if res is not None]
    failed_ids = [tid for tid, res in results_by_id.items() if res is None]

    print(f"[SUMMARY] Newly processed transcripts: {len(transcripts)}")
    print(f"[SUMMARY] Success count (this run): {len(success_ids)}")
    print(f"[SUMMARY] Failed count  (this run): {len(failed_ids)}")
    if failed_ids:
        print(f"[SUMMARY] Failed transcript ids (this run): {failed_ids}")

    # 4）追加写失败批次日志
    append_failed_records_to_jsonl(failed_records, FAILED_LOG_JSONL_PATH)

    # 5）基于完整的 success JSONL 重新生成一次 Parquet
    convert_success_jsonl_to_parquet(SUCCESS_RESULTS_JSONL_PATH, OUTPUT_PARQUET_PATH)
