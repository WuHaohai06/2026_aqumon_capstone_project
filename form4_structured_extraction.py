from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import aiohttp

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parent
DATA_FORM = "4"
DATA_DIR = REPO_ROOT / "data"
# DEFAULT_INPUT_DIR = DATA_DIR / "raw_data" / DATA_FORM
DEFAULT_INPUT_DIR = "D:\\AQUMON\\data\\golden_dataset_engine\\extracted_raw_data\\4"
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "4_extraction_v1.txt"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "results" / DATA_FORM

HK_DIFY_API_URL = "http://192.168.10.97/v1/workflows/run"
SZ_DIFY_API_URL = "http://192.168.32.50/v1/workflows/run"

GEMINI_DIFY_API_KEY = "app-mtVKZmpWyEOabRYkOR4WihPu"
GEMINI_BATCH_DIFY_API_KEY = "app-4sV7yFdvPuhiE0XSg4G4usAg"
GPT_DIFY_API_KEY = "app-SMU26wd8bd1VmfJTyhP2moe4"

MAX_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_FIRST_N_FILES: Optional[int] = None

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


@dataclass(frozen=True)
class RuntimeConfig:
    dify_profile: str
    dify_url_alias: str
    dify_api_url: str
    dify_api_key: str
    model_name: str


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "model"


def build_parser() -> argparse.ArgumentParser:
    profile_help = ", ".join(
        f"{name} -> {cfg['model_name']}" for name, cfg in sorted(DIFY_PROFILES.items())
    )
    url_help = ", ".join(
        f"{name} -> {url}" for name, url in sorted(DIFY_URL_ALIASES.items())
    )
    parser = argparse.ArgumentParser(
        description="Parse Form 4 txt filings into structured JSON files using the same Dify interface as sentiment.py."
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory containing Form 4 txt files.")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Prompt text file containing the extraction instructions.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root results directory; prompt name and model name will be appended automatically.")
    parser.add_argument("--first-n-files", type=int, default=DEFAULT_FIRST_N_FILES, help="Only process the first N pending files.")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY, help="Maximum concurrent Dify requests.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum retries per filing.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds.")
    return parser


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    dify_profile_cfg = DIFY_PROFILES[args.dify_profile]
    return RuntimeConfig(
        dify_profile=args.dify_profile,
        dify_url_alias=args.dify_url,
        dify_api_url=DIFY_URL_ALIASES[args.dify_url],
        dify_api_key=dify_profile_cfg["api_key"],
        model_name=sanitize_name(dify_profile_cfg["model_name"]),
    )


def build_runtime_paths(output_root: Path, prompt_file: Path, model_name: str) -> Dict[str, Path]:
    prompt_name = prompt_file.stem
    run_dir = output_root / f"{prompt_name}_{model_name}"
    return {
        "run_dir": run_dir,
        "structured_json_root": run_dir / "structured_json",
        "raw_response_root": run_dir / "raw_responses",
        "success_jsonl": run_dir / "structured_success.jsonl",
        "failed_jsonl": run_dir / "structured_failed.jsonl",
        "log_file": run_dir / "structured_pipeline.log",
    }


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("form4_structured_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file.parent.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_prompt_template(prompt_file_path: Path) -> str:
    if not prompt_file_path.exists() or not prompt_file_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

    prompt_text = prompt_file_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_file_path}")
    if "__FORM4_TEXT__" not in prompt_text:
        raise ValueError(f"Prompt file must contain the __FORM4_TEXT__ placeholder: {prompt_file_path}")
    return prompt_text


def read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def output_json_path(structured_json_root: Path, symbol: str, filing_id: str) -> Path:
    return structured_json_root / symbol / filing_id / f"{filing_id}.json"


def raw_response_dir(raw_response_root: Path, symbol: str, filing_id: str) -> Path:
    return raw_response_root / symbol / filing_id


def write_attempt_artifacts(
    raw_response_root: Path,
    filing: Dict[str, Any],
    attempt: int,
    *,
    status: Optional[int] = None,
    response_text: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Path:
    attempt_dir = raw_response_dir(raw_response_root, filing["symbol"], filing["id"]) / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": filing["id"],
        "symbol": filing["symbol"],
        "attempt": attempt,
        "source_file": filing["source_file"],
        "output_file": filing["output_file"],
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": utc_now_z(),
    }
    (attempt_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if response_text is not None:
        (attempt_dir / "response_body.txt").write_text(response_text, encoding="utf-8")

    return attempt_dir


def utc_now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def count_pending_form4_files(
    input_dir: Path,
    structured_json_root: Path,
    first_n_files: Optional[int] = None,
) -> int:
    if not input_dir.exists() or not input_dir.is_dir():
        return 0

    pending_count = 0
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None

    for symbol_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for filing_dir in sorted(path for path in symbol_dir.iterdir() if path.is_dir()):
            for txt_file in sorted(filing_dir.glob("*.txt")):
                filing_id = txt_file.stem
                if output_json_path(structured_json_root, symbol_dir.name, filing_id).exists():
                    continue
                pending_count += 1
                if effective_limit is not None and pending_count >= effective_limit:
                    return pending_count
    return pending_count


def iter_pending_form4_files(
    input_dir: Path,
    structured_json_root: Path,
    first_n_files: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    effective_limit = first_n_files if first_n_files is not None and first_n_files > 0 else None
    selected_count = 0

    for symbol_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for filing_dir in sorted(path for path in symbol_dir.iterdir() if path.is_dir()):
            for txt_file in sorted(filing_dir.glob("*.txt")):
                filing_id = txt_file.stem
                json_path = output_json_path(structured_json_root, symbol_dir.name, filing_id)
                if json_path.exists():
                    continue
                if effective_limit is not None and selected_count >= effective_limit:
                    return

                selected_count += 1
                yield {
                    "id": filing_id,
                    "symbol": symbol_dir.name,
                    "source_file": str(txt_file),
                    "output_file": str(json_path),
                    "text": read_text_with_fallback(txt_file),
                }


def append_success_records_to_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_failed_records_to_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json_with_auto_escape(text: str, max_fixes: int = 50) -> Any:
    working_text = text
    for _ in range(max_fixes):
        try:
            return json.loads(working_text)
        except json.JSONDecodeError as exc:
            message = str(exc)
            if "Expecting ',' delimiter" not in message and "Expecting ':' delimiter" not in message:
                raise

            error_pos = exc.pos
            index = error_pos - 1
            while index >= 0 and working_text[index] != '"':
                index -= 1
            if index < 0:
                raise
            working_text = working_text[:index] + '\\"' + working_text[index + 1 :]
    return json.loads(working_text)


def extract_json_object(answer_text: str) -> Dict[str, Any]:
    candidates: List[str] = []
    stripped = answer_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    object_match = re.search(r"\{.*\}", stripped, re.S)
    if object_match:
        candidates.append(object_match.group(0).strip())

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

        if isinstance(parsed, dict):
            return parsed

    if last_error is not None:
        raise ValueError(f"No valid JSON object found in model output: {last_error}") from last_error
    raise ValueError("No valid JSON object found in model output")


async def call_dify_for_filing(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    filing: Dict[str, Any],
    prompt_template: str,
    raw_response_root: Path,
    timeout_seconds: int,
    max_retries: int,
    runtime_config: RuntimeConfig,
    logger: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    full_prompt = prompt_template.replace("__FORM4_TEXT__", filing["text"])
    payload = {
        "inputs": {
            "input": full_prompt,
        },
        "response_mode": "blocking",
        "user": f"form4-structured-extraction-{runtime_config.model_name}",
    }
    headers = {
        "Authorization": f"Bearer {runtime_config.dify_api_key}",
        "Content-Type": "application/json",
    }

    last_error_reason: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                async with session.post(
                    runtime_config.dify_api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                ) as response:
                    status = response.status
                    response_text = await response.text()

            attempt_dir = write_attempt_artifacts(
                raw_response_root=raw_response_root,
                filing=filing,
                attempt=attempt,
                status=status,
                response_text=response_text,
            )

            if status != 200:
                last_error_reason = f"HTTP {status}: {response_text[:200]}"
                logger.error(
                    "HTTP %s on attempt %s for %s: %s [raw_response=%s]",
                    status,
                    attempt,
                    filing["id"],
                    response_text[:200],
                    attempt_dir,
                )
                if 400 <= status < 500 and status != 429:
                    break
                await asyncio.sleep(2 * attempt)
                continue

            try:
                response_payload = json.loads(response_text)
            except json.JSONDecodeError:
                last_error_reason = "Response is not valid JSON"
                logger.error(
                    "Response is not valid JSON on attempt %s for %s [raw_response=%s]",
                    attempt,
                    filing["id"],
                    attempt_dir,
                )
                await asyncio.sleep(2 * attempt)
                continue

            data = response_payload.get("data") if isinstance(response_payload, dict) else None
            outputs = data.get("outputs") if isinstance(data, dict) else None
            answer = outputs.get("text") if isinstance(outputs, dict) else None
            if not answer:
                last_error_reason = "No outputs.text in response payload"
                logger.error(
                    "No outputs.text in response payload for %s: %s [raw_response=%s]",
                    filing["id"],
                    response_payload,
                    attempt_dir,
                )
                await asyncio.sleep(2 * attempt)
                continue

            try:
                parsed_object = extract_json_object(answer)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error_reason = f"Model output is not a valid JSON object: {exc}"
                logger.error(
                    "Model output is not a valid JSON object for %s: %s [raw_response=%s]",
                    filing["id"],
                    answer[:200],
                    attempt_dir,
                )
                await asyncio.sleep(2 * attempt)
                continue

            return parsed_object, None

        except asyncio.TimeoutError:
            last_error_reason = "Timeout"
            attempt_dir = write_attempt_artifacts(
                raw_response_root=raw_response_root,
                filing=filing,
                attempt=attempt,
                error_type="Timeout",
                error_message="Request timed out before a response body was received",
            )
            logger.error("Timeout on attempt %s for %s [raw_response=%s]", attempt, filing["id"], attempt_dir)
        except aiohttp.ClientError as exc:
            last_error_reason = f"Network error: {exc}"
            attempt_dir = write_attempt_artifacts(
                raw_response_root=raw_response_root,
                filing=filing,
                attempt=attempt,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            logger.error(
                "Network error on attempt %s for %s: %s [raw_response=%s]",
                attempt,
                filing["id"],
                exc,
                attempt_dir,
            )

        await asyncio.sleep(2 * attempt)

    return None, last_error_reason or "Unknown error"


async def extract_form4_files_concurrently(
    filings: Iterable[Dict[str, Any]],
    prompt_template: str,
    raw_response_root: Path,
    success_jsonl_path: Path,
    failed_jsonl_path: Path,
    total_to_process: int,
    max_concurrency: int,
    timeout_seconds: int,
    max_retries: int,
    runtime_config: RuntimeConfig,
    logger: logging.Logger,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    sem = asyncio.Semaphore(max_concurrency)
    failed_records_all: List[Dict[str, Any]] = []

    processed_count = 0
    success_count = 0
    failed_count = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_to_process,
            desc="Form4 extraction",
            unit="file",
            dynamic_ncols=True,
        )

    async with aiohttp.ClientSession() as session:
        pending_tasks: Set[asyncio.Task] = set()
        task_meta: Dict[asyncio.Task, Dict[str, Any]] = {}

        async def process_done(done_tasks: Set[asyncio.Task]) -> None:
            nonlocal processed_count, success_count, failed_count

            for done_task in done_tasks:
                filing = task_meta.pop(done_task)
                processed_count += 1

                try:
                    parsed_object, error_reason = done_task.result()
                except Exception as exc:
                    parsed_object, error_reason = None, f"Unhandled task exception: {exc}"

                if parsed_object is None:
                    failed_count += 1
                    failed_record = {
                        "id": filing["id"],
                        "symbol": filing["symbol"],
                        "source_file": filing["source_file"],
                        "output_file": filing["output_file"],
                        "raw_response_dir": str(raw_response_dir(raw_response_root, filing["symbol"], filing["id"])),
                        "reason": error_reason or "Unknown error",
                        "attempts": max_retries,
                        "timestamp": utc_now_z(),
                    }
                    failed_records_all.append(failed_record)
                    append_failed_records_to_jsonl([failed_record], failed_jsonl_path)
                else:
                    output_file = Path(filing["output_file"])
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_text(json.dumps(parsed_object, ensure_ascii=False, indent=2), encoding="utf-8")

                    success_record = {
                        "id": filing["id"],
                        "symbol": filing["symbol"],
                        "source_file": filing["source_file"],
                        "output_file": filing["output_file"],
                        "raw_response_dir": str(raw_response_dir(raw_response_root, filing["symbol"], filing["id"])),
                        "timestamp": utc_now_z(),
                    }
                    append_success_records_to_jsonl([success_record], success_jsonl_path)
                    success_count += 1

                progress = processed_count / total_to_process if total_to_process > 0 else 1.0
                logger.info(
                    "[PROGRESS] processed=%s/%s (%.2f%%) success=%s failed=%s",
                    processed_count,
                    total_to_process,
                    progress * 100,
                    success_count,
                    failed_count,
                )

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(success=success_count, failed=failed_count)

        for filing in filings:
            task = asyncio.create_task(
                call_dify_for_filing(
                    session=session,
                    sem=sem,
                    filing=filing,
                    prompt_template=prompt_template,
                    raw_response_root=raw_response_root,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    runtime_config=runtime_config,
                    logger=logger,
                )
            )
            pending_tasks.add(task)
            task_meta[task] = filing

            if len(pending_tasks) >= max_concurrency:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                await process_done(done)

        while pending_tasks:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            await process_done(done)

    if pbar is not None:
        pbar.close()

    stats = {
        "processed": processed_count,
        "success": success_count,
        "failed": failed_count,
        "total": total_to_process,
    }
    return stats, failed_records_all


def main() -> int:
    args = build_parser().parse_args()
    runtime_config = build_runtime_config(args)
    input_dir: Path = args.input_dir
    prompt_file: Path = args.prompt_file
    output_root: Path = args.output_root
    first_n_files: Optional[int] = args.first_n_files
    max_concurrency: int = int(args.max_concurrency)
    max_retries: int = int(args.max_retries)
    timeout_seconds: int = int(args.timeout_seconds)

    runtime_paths = build_runtime_paths(output_root, prompt_file, runtime_config.model_name)
    logger = setup_logger(runtime_paths["log_file"])

    try:
        prompt_template = load_prompt_template(prompt_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    structured_json_root = runtime_paths["structured_json_root"]
    pending_total = count_pending_form4_files(
        input_dir=input_dir,
        structured_json_root=structured_json_root,
        first_n_files=first_n_files,
    )

    logger.info("Using Dify profile: %s", runtime_config.dify_profile)
    logger.info("Using Dify URL alias: %s", runtime_config.dify_url_alias)
    logger.info("Resolved Dify URL: %s", runtime_config.dify_api_url)
    logger.info("Resolved model name: %s", runtime_config.model_name)
    logger.info("Using prompt file: %s", prompt_file)
    logger.info("Output run directory: %s", runtime_paths["run_dir"])
    logger.info("Raw response root: %s", runtime_paths["raw_response_root"])
    logger.info("Pending Form 4 txt files to process: %s", pending_total)

    if pending_total <= 0:
        print("[INFO] No new Form 4 txt files to process.")
        return 0

    filings_iter = iter_pending_form4_files(
        input_dir=input_dir,
        structured_json_root=structured_json_root,
        first_n_files=first_n_files,
    )

    stats, failed_records = asyncio.run(
        extract_form4_files_concurrently(
            filings=filings_iter,
            prompt_template=prompt_template,
            raw_response_root=runtime_paths["raw_response_root"],
            success_jsonl_path=runtime_paths["success_jsonl"],
            failed_jsonl_path=runtime_paths["failed_jsonl"],
            total_to_process=pending_total,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            runtime_config=runtime_config,
            logger=logger,
        )
    )

    failed_ids = [record["id"] for record in failed_records]

    print(f"[SUMMARY] Dify profile: {runtime_config.dify_profile}")
    print(f"[SUMMARY] Dify URL   : {runtime_config.dify_url_alias} -> {runtime_config.dify_api_url}")
    print(f"[SUMMARY] Model name : {runtime_config.model_name}")
    print(f"[SUMMARY] Output dir : {runtime_paths['run_dir']}")
    print(f"[SUMMARY] Newly processed files: {stats['processed']}")
    print(f"[SUMMARY] Success count: {stats['success']}")
    print(f"[SUMMARY] Failed count: {stats['failed']}")
    if failed_ids:
        print(f"[SUMMARY] Failed ids: {failed_ids}")
    print(f"[SUMMARY] Structured JSON root: {structured_json_root}")
    print(f"[SUMMARY] Raw response root: {runtime_paths['raw_response_root']}")
    print(f"[SUMMARY] Success manifest: {runtime_paths['success_jsonl']}")
    print(f"[SUMMARY] Failed manifest: {runtime_paths['failed_jsonl']}")
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
