from __future__ import annotations

import re
from typing import Optional, Tuple


ACCESSION_YEAR_PATTERN = re.compile(r"^\d{10}-(\d{2})-\d+$")


def normalize_accession_year_range(
    start_year: Optional[int],
    end_year: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    start_yy = _normalize_single_year(start_year, "start_year")
    end_yy = _normalize_single_year(end_year, "end_year")

    if start_yy is not None and end_yy is not None and start_yy > end_yy:
        raise ValueError(
            "start_year must be less than or equal to end_year within 2000-2099."
        )

    return start_yy, end_yy


def extract_accession_year(accession_id: str) -> Optional[int]:
    match = ACCESSION_YEAR_PATTERN.match(accession_id)
    if match is None:
        return None
    return int(match.group(1))


def matches_accession_year_filter(
    accession_id: str,
    start_yy: Optional[int],
    end_yy: Optional[int],
) -> bool:
    if start_yy is None and end_yy is None:
        return True

    accession_yy = extract_accession_year(accession_id)
    if accession_yy is None:
        return False

    if start_yy is not None and accession_yy < start_yy:
        return False
    if end_yy is not None and accession_yy > end_yy:
        return False
    return True


def describe_accession_year_filter(
    start_yy: Optional[int],
    end_yy: Optional[int],
) -> str:
    if start_yy is None and end_yy is None:
        return "all years"
    if start_yy is None:
        return f"<= 20{end_yy:02d}"
    if end_yy is None:
        return f">= 20{start_yy:02d}"
    return f"20{start_yy:02d}-20{end_yy:02d} inclusive"


def _normalize_single_year(year: Optional[int], arg_name: str) -> Optional[int]:
    if year is None:
        return None
    if not 2000 <= int(year) <= 2099:
        raise ValueError(f"{arg_name} must be between 2000 and 2099, got {year}.")
    return int(year) % 100
