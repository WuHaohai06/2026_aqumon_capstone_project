import aiohttp
import asyncio
import json
import re
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


DIFY_API_URL = "http://192.168.10.97/v1/workflows/run"
DIFY_API_KEY = "app-mtVKZmpWyEOabRYkOR4WihPu"

ROOT_DIR = "D:\\AQUMON"
DATA_FORM = "4"

DATA_DIR = Path(ROOT_DIR) / "data"
STRUCTURED_INPUT_DIR = DATA_DIR / "results" / DATA_FORM / "4_extraction_v1" / "structured_json"

PROMPTS_DIR = Path(ROOT_DIR) / "prompts"
PROMPT_FILE_PATH = PROMPTS_DIR / "4_structured_input.txt"
PROMPT_NAME = PROMPT_FILE_PATH.stem

MAX_CONCURRENCY = 8
MAX_RETRIES = 1
TEST_FIRST_N_FILES = None

RESULTS_BASE_DIR = DATA_DIR / "results" / DATA_FORM / PROMPT_NAME
SUCCESS_RESULTS_JSONL_PATH = str(RESULTS_BASE_DIR / "sentiment_results.jsonl")
FAILED_LOG_JSONL_PATH = str(RESULTS_BASE_DIR / "failed_transcripts.jsonl")
OUTPUT_PARQUET_PATH = str(RESULTS_BASE_DIR / "sentiment_results.parquet")
LOG_FILE_PATH = str(RESULTS_BASE_DIR / "sentiment_pipeline.log")


logger = logging.getLogger("sentiment_structured_pipeline")
logger.setLevel(logging.INFO)

RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


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


def load_processed_ids_from_jsonl(path: str) -> Set[str]:
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
                filing_id = obj.get("id")
                if filing_id:
                    processed_ids.add(filing_id)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %s already-processed ids from %s", len(processed_ids), path)
    return processed_ids


def count_pending_structured_filings(
    dir_path: Path,
    skip_ids: Set[str],
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
    batch_size: int = 5,
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
    path: str,
) -> None:
    if not batch_result:
        return
    with open(path, "a", encoding="utf-8") as f:
        for item in batch_result:
            if "id" not in item:
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_failed_records_to_jsonl(
    failed_records: List[Dict[str, Any]],
    path: str,
) -> None:
    if not failed_records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_json_with_auto_escape(s: str, max_fixes: int = 50):
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
) -> Optional[List[Dict[str, Any]]]:
    filings_payload = [
        {"id": item["id"], "filing": item["filing"]}
        for item in batch
    ]
    filings_json_str = json.dumps(filings_payload, ensure_ascii=False)
    full_prompt = prompt_template.replace("__FILINGS_JSON__", filings_json_str)

    payload = {
        "inputs": {"input": full_prompt},
        "response_mode": "blocking",
        "user": "form4-structured-sentiment-batch",
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
                    timeout=120,
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
                except json.JSONDecodeError:
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
    logger.error("Batch with ids %s failed after %s attempts.", batch_ids, MAX_RETRIES)
    failed_records.append(
        {
            "batch_ids": batch_ids,
            "reason": last_error_reason or "Unknown error",
            "attempts": MAX_RETRIES,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )
    return None


async def analyze_transcripts_concurrently(
    transcripts: Iterable[Dict[str, Any]],
    prompt_template: str,
    total_to_process: int,
    max_concurrency: int = MAX_CONCURRENCY,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    failed_records_all: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_concurrency)

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

                try:
                    batch_result = done_task.result()
                except Exception as e:
                    batch_result = None
                    failed_records_all.append(
                        {
                            "batch_ids": batch_ids,
                            "reason": f"Unhandled task exception: {e}",
                            "attempts": MAX_RETRIES,
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

                append_success_results_to_jsonl(batch_result, SUCCESS_RESULTS_JSONL_PATH)
                success_count += len(batch_result)
                failed_count += max(0, batch_size - len(batch_result))

                if pbar is not None:
                    pbar.set_postfix(success=success_count, failed=failed_count)

        for batch in chunk_transcripts(transcripts, batch_size=5):
            task = asyncio.create_task(
                call_dify_batch(session, sem, batch, failed_records_all, prompt_template)
            )
            pending_tasks.add(task)
            task_meta[task] = {
                "size": len(batch),
                "ids": [item["id"] for item in batch],
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


def convert_success_jsonl_to_parquet(
    success_jsonl_path: str,
    parquet_path: str,
) -> None:
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

            reasons = obj.get("reasons", {}) or {}
            row = {
                "id": obj.get("id"),
                "sentiment_score": obj.get("sentiment_score"),
                "confidence": obj.get("confidence"),
                "summary": obj.get("summary"),
                "reasons_positives": json.dumps(reasons.get("positives", []), ensure_ascii=False),
                "reasons_negatives": json.dumps(reasons.get("negatives", []), ensure_ascii=False),
                "reasons_guidance_tone": reasons.get("guidance_tone", ""),
                "reasons_qa_tone": reasons.get("qa_tone", ""),
            }
            rows.append(row)

    if not rows:
        print("[INFO] No valid rows parsed from success JSONL, skip Parquet.")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Converted {len(rows)} rows from JSONL to Parquet: {parquet_path}")


if __name__ == "__main__":
    prompt_template = load_prompt_template(PROMPT_FILE_PATH)

    processed_ids = load_processed_ids_from_jsonl(SUCCESS_RESULTS_JSONL_PATH)

    pending_total = count_pending_structured_filings(
        STRUCTURED_INPUT_DIR,
        processed_ids,
        first_n_files=TEST_FIRST_N_FILES,
    )

    if pending_total <= 0:
        print("[INFO] No new structured filings to process. You can still regenerate Parquet from existing JSONL.")
        convert_success_jsonl_to_parquet(SUCCESS_RESULTS_JSONL_PATH, OUTPUT_PARQUET_PATH)
        sys.exit(0)

    transcripts_iter = iter_structured_filings(
        STRUCTURED_INPUT_DIR,
        processed_ids,
        first_n_files=TEST_FIRST_N_FILES,
    )

    stats, failed_records = asyncio.run(
        analyze_transcripts_concurrently(
            transcripts_iter,
            prompt_template,
            total_to_process=pending_total,
        )
    )

    failed_ids = [filing_id for rec in failed_records for filing_id in rec.get("batch_ids", [])]

    print(f"[SUMMARY] Newly processed transcripts: {stats['processed']}")
    print(f"[SUMMARY] Success count (this run): {stats['success']}")
    print(f"[SUMMARY] Failed count  (this run): {stats['failed']}")
    if failed_ids:
        print(f"[SUMMARY] Failed transcript ids (this run): {failed_ids}")

    append_failed_records_to_jsonl(failed_records, FAILED_LOG_JSONL_PATH)
    convert_success_jsonl_to_parquet(SUCCESS_RESULTS_JSONL_PATH, OUTPUT_PARQUET_PATH)