import argparse
import aiohttp
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd

from accession_year_filter import (
    describe_accession_year_filter,
    matches_accession_year_filter,
    normalize_accession_year_range,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parent
DATA_FORM = "4"
DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "prompts"

HK_DIFY_API_URL = "http://192.168.10.97/v1/workflows/run"
SZ_DIFY_API_URL = "http://192.168.32.50/v1/workflows/run"
GEMINI_DIFY_API_KEY = "app-mtVKZmpWyEOabRYkOR4WihPu"
GEMINI_BATCH_DIFY_API_KEY = "app-4sV7yFdvPuhiE0XSg4G4usAg"
GPT_DIFY_API_KEY = "app-SMU26wd8bd1VmfJTyhP2moe4"

DEFAULT_INPUT_DIR = DATA_DIR / "results" / DATA_FORM / "4_extraction_v1" / "structured_json"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "4_golden_structured_input.txt"
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 1
DEFAULT_BATCH_SIZE = 3
DEFAULT_TIMEOUT_SECONDS = 120

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

logger = logging.getLogger("sentiment_structured_pipeline")


@dataclass(frozen=True)
class RuntimeConfig:
    dify_profile: str
    dify_url_alias: str
    dify_api_url: str
    dify_api_key: str
    model_name: str
    input_dir: Path
    prompt_file: Path
    prompt_name: str
    result_dir_name: str
    results_base_dir: Path
    completion_markers_root: Path
    success_jsonl_path: Path
    failed_log_path: Path
    output_parquet_path: Path
    log_file_path: Path
    max_concurrency: int
    batch_size: int
    max_retries: int
    first_n_files: Optional[int]
    reprocess_existing: bool


def parse_args() -> argparse.Namespace:
    profile_help = ", ".join(
        f"{name} -> {cfg['model_name']}" for name, cfg in sorted(DIFY_PROFILES.items())
    )
    url_help = ", ".join(
        f"{name} -> {url}" for name, url in sorted(DIFY_URL_ALIASES.items())
    )

    parser = argparse.ArgumentParser(
        description="Run structured Form 4 sentiment analysis from structured JSON inputs.",
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
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt template file. Must contain __FILINGS_JSON__.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing structured JSON filings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "data/results/4/<prompt_name>_<model_name>."
        ),
    )
    parser.add_argument(
        "--first-n-files",
        type=int,
        default=None,
        help="Only process the first N pending filings.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Inclusive accession year filter start in YYYY, e.g. 2015.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Inclusive accession year filter end in YYYY, e.g. 2020.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum number of concurrent Dify requests. Default: {DEFAULT_MAX_CONCURRENCY}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of filings sent in each Dify batch. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts per batch. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Ignore existing success JSONL and reprocess all filings into the selected output folder.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "model"


def configure_logger(log_file_path: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

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


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    dify_profile_cfg = DIFY_PROFILES[args.dify_profile]
    dify_api_url = DIFY_URL_ALIASES[args.dify_url]
    prompt_file = args.prompt_file.resolve()
    input_dir = args.input_dir.resolve()
    prompt_name = prompt_file.stem
    model_name = sanitize_name(dify_profile_cfg["model_name"])
    result_dir_name = f"{prompt_name}_{model_name}"

    results_base_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DATA_DIR / "results" / DATA_FORM / result_dir_name
    )

    return RuntimeConfig(
        dify_profile=args.dify_profile,
        dify_url_alias=args.dify_url,
        dify_api_url=dify_api_url,
        dify_api_key=dify_profile_cfg["api_key"],
        model_name=model_name,
        input_dir=input_dir,
        prompt_file=prompt_file,
        prompt_name=prompt_name,
        result_dir_name=result_dir_name,
        results_base_dir=results_base_dir,
        completion_markers_root=results_base_dir / "completion_markers",
        success_jsonl_path=results_base_dir / "sentiment_results.jsonl",
        failed_log_path=results_base_dir / "failed_transcripts.jsonl",
        output_parquet_path=results_base_dir / "sentiment_results.parquet",
        log_file_path=results_base_dir / "sentiment_pipeline.log",
        max_concurrency=args.max_concurrency,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        first_n_files=args.first_n_files,
        reprocess_existing=args.reprocess_existing,
    )


def load_prompt_template(prompt_file_path: Path) -> str:
    if prompt_file_path.exists() and prompt_file_path.is_file():
        prompt_text = prompt_file_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt file is empty: {prompt_file_path}")
        if "__FILINGS_JSON__" not in prompt_text:
            raise ValueError(f"Prompt file must contain __FILINGS_JSON__: {prompt_file_path}")
        logger.info("Using prompt file: %s", prompt_file_path)
        return prompt_text
    raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")


def completion_marker_path(markers_root: Path, filing_id: str) -> Path:
    return markers_root / f"{filing_id}.done"


def write_completion_marker(
    markers_root: Path,
    filing_id: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    marker_path = completion_marker_path(markers_root, filing_id)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": filing_id}
    if metadata:
        payload.update(metadata)
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return marker_path


def load_processed_ids_from_markers(markers_root: Path) -> Set[str]:
    if not markers_root.exists():
        return set()
    return {marker.stem for marker in markers_root.glob("*.done")}


def load_processed_ids_from_jsonl(path: Path) -> Set[str]:
    processed_ids: Set[str] = set()
    if not path.exists():
        return processed_ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                filing_id = obj.get("id")
                if filing_id:
                    processed_ids.add(filing_id)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %s already-processed ids from %s", len(processed_ids), path)
    return processed_ids


def ensure_completion_markers_from_existing_results(
    markers_root: Path,
    success_jsonl_path: Path,
) -> Set[str]:
    marker_ids = load_processed_ids_from_markers(markers_root)
    legacy_ids = load_processed_ids_from_jsonl(success_jsonl_path)
    missing_ids = legacy_ids - marker_ids

    for filing_id in sorted(missing_ids):
        write_completion_marker(
            markers_root,
            filing_id,
            metadata={
                "created_from": "legacy_success_jsonl",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    if missing_ids:
        logger.info(
            "Created %s missing completion markers from existing success JSONL.",
            len(missing_ids),
        )

    return marker_ids | missing_ids


def count_pending_structured_filings(
    dir_path: Path,
    skip_ids: Set[str],
    start_year_yy: Optional[int] = None,
    end_year_yy: Optional[int] = None,
    first_n_files: Optional[int] = None,
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("Directory not found or not a directory: %s", dir_path)
        return 0

    symbol_dirs = sorted([p for p in dir_path.iterdir() if p.is_dir()])
    if not symbol_dirs:
        logger.warning("No symbol directories found under: %s", dir_path)

    pending_count = 0
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None

    for symbol_dir in symbol_dirs:
        id_dirs = sorted([p for p in symbol_dir.iterdir() if p.is_dir()])
        for id_dir in id_dirs:
            json_files = sorted(id_dir.glob("*.json"))
            for fp in json_files:
                filing_id = fp.stem
                if not matches_accession_year_filter(filing_id, start_year_yy, end_year_yy):
                    continue
                if filing_id in skip_ids:
                    continue
                pending_count += 1
                if effective_limit is not None and pending_count >= effective_limit:
                    logger.info(
                        "[TEST MODE] first_n_files=%s, pending=%s, reached_limit=True",
                        effective_limit,
                        pending_count,
                    )
                    return pending_count

    logger.info("Pending structured filings to process: %s", pending_count)
    if effective_limit is not None:
        logger.info(
            "[TEST MODE] first_n_files=%s, pending=%s, reached_limit=False",
            effective_limit,
            pending_count,
        )
    return pending_count


def iter_structured_filings(
    dir_path: Path,
    skip_ids: Set[str],
    start_year_yy: Optional[int] = None,
    end_year_yy: Optional[int] = None,
    first_n_files: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("Directory not found or not a directory: %s", dir_path)
        return

    symbol_dirs = sorted([p for p in dir_path.iterdir() if p.is_dir()])
    if not symbol_dirs:
        logger.warning("No symbol directories found under: %s", dir_path)
        return

    selected_count = 0
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None

    for symbol_dir in symbol_dirs:
        id_dirs = sorted([p for p in symbol_dir.iterdir() if p.is_dir()])
        for id_dir in id_dirs:
            json_files = sorted(id_dir.glob("*.json"))
            for fp in json_files:
                filing_id = fp.stem
                if not matches_accession_year_filter(filing_id, start_year_yy, end_year_yy):
                    continue
                if filing_id in skip_ids:
                    continue
                if effective_limit is not None and selected_count >= effective_limit:
                    return

                structured_filing = json.loads(fp.read_text(encoding="utf-8"))
                selected_count += 1
                yield {
                    "id": filing_id,
                    "filing": structured_filing,
                    "symbol": symbol_dir.name,
                    "source_file": str(fp),
                }


def chunk_transcripts(
    transcripts: Iterable[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in transcripts:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def append_success_results_to_jsonl(
    batch_result: List[Dict[str, Any]],
    path: Path,
    *,
    allowed_ids: Optional[Set[str]] = None,
) -> List[str]:
    written_ids: List[str] = []
    if not batch_result:
        return written_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in batch_result:
            filing_id = item.get("id")
            if not filing_id:
                continue
            if allowed_ids is not None and filing_id not in allowed_ids:
                logger.warning("Skip writing unexpected id not present in input batch: %s", filing_id)
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written_ids.append(filing_id)
    return written_ids


def append_failed_records_to_jsonl(failed_records: List[Dict[str, Any]], path: Path) -> None:
    if not failed_records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json_with_auto_escape(s: str, max_fixes: int = 50) -> Any:
    text = s
    for _ in range(max_fixes):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            msg = str(e)
            if "Expecting ',' delimiter" not in msg and "Expecting ':' delimiter" not in msg:
                raise

            error_pos = e.pos
            i = error_pos - 1
            while i >= 0 and text[i] != '"':
                i -= 1

            if i < 0:
                raise

            text = text[:i] + '\\"' + text[i + 1 :]
    return json.loads(text)


def extract_json_array(answer_text: str) -> List[Dict[str, Any]]:
    candidates: List[str] = []
    stripped = answer_text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", stripped, re.S)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    array_match = re.search(r"\[\s*\{.*\}\s*\]", stripped, re.S)
    if array_match:
        candidates.append(array_match.group(0).strip())

    if stripped:
        candidates.append(stripped)

    seen = set()
    last_error: Optional[Exception] = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = load_json_with_auto_escape(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue

        if isinstance(parsed, list):
            return parsed

    if last_error is not None:
        raise ValueError(f"No valid JSON array found in model output: {last_error}") from last_error
    raise ValueError("No valid JSON array found in model output")


async def call_dify_batch(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    batch: List[Dict[str, Any]],
    failed_records: List[Dict[str, Any]],
    prompt_template: str,
    config: RuntimeConfig,
) -> Optional[List[Dict[str, Any]]]:
    filings_payload = [{"id": item["id"], "filing": item["filing"]} for item in batch]
    filings_json_str = json.dumps(filings_payload, ensure_ascii=False)
    full_prompt = prompt_template.replace("__FILINGS_JSON__", filings_json_str)

    payload = {
        "inputs": {"input": full_prompt},
        "response_mode": "blocking",
        "user": f"form4-structured-sentiment-batch-{config.model_name}",
    }

    headers = {
        "Authorization": f"Bearer {config.dify_api_key}",
        "Content-Type": "application/json",
    }

    last_error_reason = None

    for attempt in range(1, config.max_retries + 1):
        try:
            async with sem:
                async with session.post(
                    config.dify_api_url,
                    headers=headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                ) as resp:
                    status = resp.status
                    text = await resp.text()

            if status != 200:
                last_error_reason = f"HTTP {status}: {text[:200]}"
                logger.error("HTTP %s on attempt %s: %s", status, attempt, text[:200])
                if 400 <= status < 500 and status != 429:
                    break
            else:
                try:
                    data = json.loads(text)["data"]
                except (json.JSONDecodeError, KeyError):
                    last_error_reason = "Response is not valid JSON"
                    logger.error("Response is not valid JSON on attempt %s", attempt)
                    await asyncio.sleep(2 * attempt)
                    continue

                answer = data.get("outputs", {}).get("text")
                if not answer:
                    last_error_reason = "No 'text' field in response outputs"
                    logger.error("No 'text' field in response: %s", data)
                    await asyncio.sleep(2 * attempt)
                    continue

                try:
                    result_json_array = extract_json_array(answer)
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error_reason = f"'answer' field is not valid JSON array: {exc}"
                    logger.error("'answer' field is not valid JSON array: %s", str(exc))
                    await asyncio.sleep(2 * attempt)
                    continue

                if not isinstance(result_json_array, list):
                    last_error_reason = "Parsed answer is not a list"
                    logger.error("Parsed answer is not a list: %s", result_json_array)
                    await asyncio.sleep(2 * attempt)
                    continue

                if len(result_json_array) != len(batch):
                    logger.warning(
                        "Result length %s != batch length %s",
                        len(result_json_array),
                        len(batch),
                    )

                return result_json_array

        except asyncio.TimeoutError:
            last_error_reason = "Timeout"
            logger.error("Timeout on attempt %s", attempt)
        except aiohttp.ClientError as e:
            last_error_reason = f"Network error: {e}"
            logger.error("Network error on attempt %s: %s", attempt, e)

        await asyncio.sleep(2 * attempt)

    batch_ids = [item["id"] for item in batch]
    logger.error("Batch with ids %s failed after %s attempts.", batch_ids, config.max_retries)
    failed_records.append(
        {
            "batch_ids": batch_ids,
            "reason": last_error_reason or "Unknown error",
            "attempts": config.max_retries,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )
    return None


async def analyze_transcripts_concurrently(
    transcripts: Iterable[Dict[str, Any]],
    prompt_template: str,
    total_to_process: int,
    config: RuntimeConfig,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    failed_records_all: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(config.max_concurrency)

    processed_count = 0
    success_count = 0
    failed_count = 0

    logger.info("Start processing %s structured filings in this run.", total_to_process)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_to_process,
            desc="Structured sentiment batches",
            unit="filing",
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
                items_by_id = meta.get("items_by_id", {})

                try:
                    batch_result = done_task.result()
                except Exception as e:
                    batch_result = None
                    failed_records_all.append(
                        {
                            "batch_ids": batch_ids,
                            "reason": f"Unhandled task exception: {e}",
                            "attempts": config.max_retries,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                    )

                processed_count += batch_size
                remaining = total_to_process - processed_count
                progress = processed_count / total_to_process if total_to_process > 0 else 1.0

                logger.info(
                    "[PROGRESS] Processed %s/%s filings (%.2f%%), remaining %s",
                    processed_count,
                    total_to_process,
                    progress * 100,
                    remaining,
                )

                if pbar is not None:
                    pbar.update(batch_size)

                if batch_result is None:
                    failed_count += batch_size
                    if pbar is not None:
                        pbar.set_postfix(success=success_count, failed=failed_count)
                    continue

                written_ids = append_success_results_to_jsonl(
                    batch_result,
                    config.success_jsonl_path,
                    allowed_ids=set(batch_ids),
                )

                for filing_id in written_ids:
                    source_item = items_by_id.get(filing_id, {})
                    write_completion_marker(
                        config.completion_markers_root,
                        filing_id,
                        metadata={
                            "symbol": source_item.get("symbol"),
                            "source_file": source_item.get("source_file"),
                            "result_dir": str(config.results_base_dir),
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        },
                    )

                success_count += len(written_ids)
                failed_count += max(0, batch_size - len(written_ids))

                if pbar is not None:
                    pbar.set_postfix(success=success_count, failed=failed_count)

        for batch in chunk_transcripts(transcripts, batch_size=config.batch_size):
            task = asyncio.create_task(
                call_dify_batch(session, sem, batch, failed_records_all, prompt_template, config)
            )
            pending_tasks.add(task)
            task_meta[task] = {
                "size": len(batch),
                "ids": [item["id"] for item in batch],
                "items_by_id": {item["id"]: item for item in batch},
            }

            if len(pending_tasks) >= config.max_concurrency:
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


def convert_success_jsonl_to_parquet(success_jsonl_path: Path, parquet_path: Path) -> None:
    if not success_jsonl_path.exists():
        print(f"[INFO] No success JSONL found at {success_jsonl_path}, skip Parquet conversion.")
        return

    rows = []
    with success_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            reasons = obj.get("reasons", {}) or {}
            signal_breakdown = obj.get("signal_breakdown", {}) or {}
            row = {
                "id": obj.get("id"),
                "sentiment_score": obj.get("sentiment_score"),
                "confidence": obj.get("confidence"),
                "summary": obj.get("summary"),
                "signal_direction": signal_breakdown.get("direction", ""),
                "signal_insider_role_strength": signal_breakdown.get("insider_role_strength", ""),
                "signal_size_assessment": signal_breakdown.get("size_assessment", ""),
                "signal_discretionary_level": signal_breakdown.get("discretionary_level", ""),
                "signal_ownership_type": signal_breakdown.get("ownership_type", ""),
                "signal_information_quality": signal_breakdown.get("information_quality", ""),
                "decision_factors": json.dumps(obj.get("decision_factors", []), ensure_ascii=False),
                "risk_flags": json.dumps(obj.get("risk_flags", []), ensure_ascii=False),
                "reasons_positives": json.dumps(reasons.get("positives", []), ensure_ascii=False),
                "reasons_negatives": json.dumps(reasons.get("negatives", []), ensure_ascii=False),
                "reasons_guidance_tone": reasons.get("guidance_tone", ""),
                "reasons_qa_tone": reasons.get("qa_tone", ""),
            }
            rows.append(row)

    if not rows:
        print("[INFO] No valid rows parsed from success JSONL, skip Parquet.")
        return

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Converted {len(rows)} rows from JSONL to Parquet: {parquet_path}")


def main() -> int:
    args = parse_args()
    try:
        start_year_yy, end_year_yy = normalize_accession_year_range(args.start_year, args.end_year)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    config = build_runtime_config(args)

    config.results_base_dir.mkdir(parents=True, exist_ok=True)
    configure_logger(config.log_file_path)

    logger.info("Using Dify profile: %s", config.dify_profile)
    logger.info("Using Dify URL alias: %s", config.dify_url_alias)
    logger.info("Resolved Dify URL: %s", config.dify_api_url)
    logger.info("Resolved model name: %s", config.model_name)
    logger.info("Result directory: %s", config.results_base_dir)
    logger.info("Completion markers: %s", config.completion_markers_root)
    logger.info("Input directory: %s", config.input_dir)
    logger.info("Prompt file: %s", config.prompt_file)
    logger.info("Max concurrency: %s", config.max_concurrency)
    logger.info("Batch size: %s", config.batch_size)
    logger.info("Accession year filter: %s", describe_accession_year_filter(start_year_yy, end_year_yy))

    prompt_template = load_prompt_template(config.prompt_file)

    processed_ids = (
        set()
        if config.reprocess_existing
        else ensure_completion_markers_from_existing_results(
            config.completion_markers_root,
            config.success_jsonl_path,
        )
    )

    pending_total = count_pending_structured_filings(
        config.input_dir,
        processed_ids,
        start_year_yy=start_year_yy,
        end_year_yy=end_year_yy,
        first_n_files=config.first_n_files,
    )

    if pending_total <= 0:
        print("[INFO] No new structured filings to process. You can still regenerate Parquet from existing JSONL.")
        convert_success_jsonl_to_parquet(config.success_jsonl_path, config.output_parquet_path)
        return 0

    transcripts_iter = iter_structured_filings(
        config.input_dir,
        processed_ids,
        start_year_yy=start_year_yy,
        end_year_yy=end_year_yy,
        first_n_files=config.first_n_files,
    )

    stats, failed_records = asyncio.run(
        analyze_transcripts_concurrently(
            transcripts_iter,
            prompt_template,
            total_to_process=pending_total,
            config=config,
        )
    )

    failed_ids = [filing_id for rec in failed_records for filing_id in rec.get("batch_ids", [])]

    print(f"[SUMMARY] Dify profile: {config.dify_profile}")
    print(f"[SUMMARY] Dify URL   : {config.dify_url_alias} -> {config.dify_api_url}")
    print(f"[SUMMARY] Model name : {config.model_name}")
    print(f"[SUMMARY] Output dir : {config.results_base_dir}")
    print(f"[SUMMARY] Completion markers: {config.completion_markers_root}")
    print(f"[SUMMARY] Max concurrency: {config.max_concurrency}")
    print(f"[SUMMARY] Batch size: {config.batch_size}")
    print(f"[SUMMARY] Accession year filter: {describe_accession_year_filter(start_year_yy, end_year_yy)}")
    print(f"[SUMMARY] Newly processed transcripts: {stats['processed']}")
    print(f"[SUMMARY] Success count (this run): {stats['success']}")
    print(f"[SUMMARY] Failed count  (this run): {stats['failed']}")
    if failed_ids:
        print(f"[SUMMARY] Failed transcript ids (this run): {failed_ids}")

    append_failed_records_to_jsonl(failed_records, config.failed_log_path)
    convert_success_jsonl_to_parquet(config.success_jsonl_path, config.output_parquet_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
