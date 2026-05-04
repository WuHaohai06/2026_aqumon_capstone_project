from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ACCESSION_DASH_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
ACCESSION_NODASH_RE = re.compile(r"^\d{18}$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_accession(value: object) -> Optional[str]:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return None
    if ACCESSION_DASH_RE.fullmatch(text):
        return text

    digits = re.sub(r"\D", "", text)
    if ACCESSION_NODASH_RE.fullmatch(digits):
        return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return None


def extract_accession(values: Iterable[object]) -> Optional[str]:
    for value in values:
        accession = normalize_accession(value)
        if accession:
            return accession
    return None


def build_row_mapping(fieldnames: Sequence[str], row: Sequence[str]) -> Dict[str, str]:
    width = max(len(fieldnames), len(row))
    mapping: Dict[str, str] = {}
    for index in range(width):
        if index < len(fieldnames) and fieldnames[index].strip():
            key = fieldnames[index].strip()
        else:
            key = f"column_{index + 1}"
        mapping[key] = row[index].strip() if index < len(row) else ""
    return mapping


def load_accession_records(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records: List[Dict[str, object]] = []
    seen = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return records

    header = [cell.strip() for cell in rows[0]]
    has_header = any("accession" in cell.lower() for cell in header)

    if has_header:
        fieldnames = header
        data_rows = rows[1:]
        start_row_number = 2
    else:
        fieldnames = [f"column_{index + 1}" for index in range(len(rows[0]))]
        data_rows = rows
        start_row_number = 1

    for offset, row in enumerate(data_rows):
        row_number = start_row_number + offset
        row_mapping = build_row_mapping(fieldnames, row)
        accession = extract_accession(row_mapping.values())
        if not accession or accession in seen:
            continue
        seen.add(accession)
        records.append(
            {
                "row_number": row_number,
                "accession": accession,
                "source_fields": row_mapping,
            }
        )
    return records


def load_source_dirs_from_metadata(
    raw_root: Path,
    metadata_csv: Path,
    accessions: Sequence[str],
) -> Dict[str, List[Path]]:
    results: Dict[str, List[Path]] = {}
    pending = set(accessions)
    if not pending or not metadata_csv.exists():
        return results

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = normalize_accession(row.get("accessionNumber"))
            if accession not in pending:
                continue

            form = (row.get("form") or "").strip()
            ticker = (row.get("ticker") or "").strip().upper()
            if not form or not ticker:
                continue

            source_dir = raw_root / form / ticker / accession
            if not source_dir.is_dir():
                continue

            results.setdefault(accession, [])
            if source_dir not in results[accession]:
                results[accession].append(source_dir)
            pending.discard(accession)
            if not pending:
                break
    return results


def load_source_dirs_from_glob(
    raw_root: Path,
    accessions: Iterable[str],
    existing: Dict[str, List[Path]],
) -> Dict[str, List[Path]]:
    for accession in accessions:
        if accession in existing:
            continue
        matches = [path for path in raw_root.glob(f"*/*/{accession}") if path.is_dir()]
        if matches:
            existing[accession] = sorted(matches, key=lambda path: str(path))
    return existing


def iter_copyable_files(source_dir: Path) -> Iterable[Path]:
    for child in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not child.is_file():
            continue
        if child.name.startswith("."):
            continue
        yield child


def copy_source_dir(source_dir: Path, raw_root: Path, target_root: Path) -> Dict[str, object]:
    relative_dir = source_dir.relative_to(raw_root)
    destination_dir = target_root / relative_dir

    copied_files = 0
    skipped_existing = 0
    for source_file in iter_copyable_files(source_dir):
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_file = destination_dir / source_file.name
        if destination_file.exists() and destination_file.stat().st_size == source_file.stat().st_size:
            skipped_existing += 1
            continue
        shutil.copy2(source_file, destination_file)
        copied_files += 1

    return {
        "destination_dir": destination_dir,
        "copied_files": copied_files,
        "skipped_existing": skipped_existing,
    }


def write_manifest(manifest_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "accession",
        "status",
        "source_dir",
        "destination_dir",
        "copied_files",
        "skipped_existing",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_source_fieldnames(records: Sequence[Dict[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    seen = set()
    for record in records:
        source_fields = record.get("source_fields", {})
        if not isinstance(source_fields, dict):
            continue
        for key in source_fields.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def write_missing_report(
    missing_report_path: Path,
    missing_rows: Sequence[Dict[str, object]],
    source_fieldnames: Sequence[str],
) -> None:
    missing_report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_number", "accession", "reason", *source_fieldnames]
    with missing_report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)


def build_parser() -> argparse.ArgumentParser:
    default_csv = REPO_ROOT / "data" / "golden_dataset_engine" / "golden_data_accession_number.csv"
    default_raw_root = REPO_ROOT / "data" / "raw_data"
    default_target_root = REPO_ROOT / "data" / "golden_dataset_engine" / "extracted_raw_data"
    default_manifest = REPO_ROOT / "data" / "golden_dataset_engine" / "extraction_manifest.csv"
    default_missing_report = REPO_ROOT / "data" / "golden_dataset_engine" / "missing_file_info.csv"

    parser = argparse.ArgumentParser(
        description="Extract raw filing files listed by accession number into the golden dataset directory."
    )
    parser.add_argument("--csv", type=Path, default=default_csv, help="CSV containing accession numbers.")
    parser.add_argument("--raw-root", type=Path, default=default_raw_root, help="Root raw_data directory.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=default_target_root,
        help="Destination root for extracted files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Path to write the extraction manifest CSV.",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=default_missing_report,
        help="Path to write a CSV containing missing file information.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_path: Path = args.csv
    raw_root: Path = args.raw_root
    target_root: Path = args.target_root
    manifest_path: Path = args.manifest
    missing_report_path: Path = args.missing_report

    records = load_accession_records(csv_path)
    accessions = [str(record["accession"]) for record in records]
    source_fieldnames = collect_source_fieldnames(records)
    print(f"Loaded {len(accessions)} accession numbers from {csv_path}")

    manifest_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    if not accessions:
        write_manifest(manifest_path, manifest_rows)
        write_missing_report(missing_report_path, missing_rows, source_fieldnames)
        print("No accession numbers were found in the CSV. Nothing to extract.")
        print(f"Manifest written to {manifest_path}")
        print(f"Missing report written to {missing_report_path}")
        return 0

    metadata_csv = raw_root / "filing_metadata.csv"
    source_dirs = load_source_dirs_from_metadata(raw_root, metadata_csv, accessions)
    source_dirs = load_source_dirs_from_glob(raw_root, accessions, source_dirs)

    copied_file_total = 0
    skipped_file_total = 0
    found_accession_count = 0

    for record in records:
        accession = str(record["accession"])
        matches = source_dirs.get(accession, [])
        if not matches:
            manifest_rows.append(
                {
                    "accession": accession,
                    "status": "missing",
                    "source_dir": "",
                    "destination_dir": "",
                    "copied_files": 0,
                    "skipped_existing": 0,
                }
            )
            missing_row = {
                "row_number": record.get("row_number", ""),
                "accession": accession,
                "reason": "not found under raw_data",
            }
            source_fields = record.get("source_fields", {})
            if isinstance(source_fields, dict):
                for key, value in source_fields.items():
                    missing_row[key] = value
            missing_rows.append(missing_row)
            continue

        found_accession_count += 1
        for source_dir in matches:
            copy_result = copy_source_dir(source_dir, raw_root, target_root)
            copied_file_total += int(copy_result["copied_files"])
            skipped_file_total += int(copy_result["skipped_existing"])
            manifest_rows.append(
                {
                    "accession": accession,
                    "status": "copied",
                    "source_dir": str(source_dir),
                    "destination_dir": str(copy_result["destination_dir"]),
                    "copied_files": copy_result["copied_files"],
                    "skipped_existing": copy_result["skipped_existing"],
                }
            )

    write_manifest(manifest_path, manifest_rows)
    write_missing_report(missing_report_path, missing_rows, source_fieldnames)

    print(f"Matched {found_accession_count}/{len(accessions)} accession numbers")
    print(f"Copied {copied_file_total} files into {target_root}")
    print(f"Skipped {skipped_file_total} existing files")
    print(f"Manifest written to {manifest_path}")
    print(f"Missing report written to {missing_report_path}")
    if missing_rows:
        print(f"Missing {len(missing_rows)} accession numbers:")
        for missing_row in missing_rows:
            ticker = str(missing_row.get("Ticker") or missing_row.get("ticker") or "").strip()
            prefix = f" - {ticker} | " if ticker else " - "
            print(f"{prefix}{missing_row['accession']} | row {missing_row['row_number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
