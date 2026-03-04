"""
Download SEC filing metadata into a CSV file under the output root.
"""
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class SecClient:
    def __init__(self, user_agent, min_delay=0.2, max_retries=5, timeout=30):
        self.headers = {"User-Agent": user_agent}
        self.min_delay = float(min_delay)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self._last_request_ts = 0.0

    def _sleep_if_needed(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

    def get(self, url, timeout=None):
        for attempt in range(self.max_retries):
            self._sleep_if_needed()
            try:
                response = requests.get(url, headers=self.headers, timeout=timeout or self.timeout)
                self._last_request_ts = time.time()
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.min_delay * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.min_delay * (2 ** attempt))
        raise RuntimeError(f"Failed to GET {url}")

    def get_json(self, url, timeout=None):
        return self.get(url, timeout=timeout).json()


class _SimpleProgress:
    def __init__(self, total=None, desc=""):
        self.total = total
        self.desc = desc
        self.count = 0
        self._last_len = 0

    def update(self, n=1):
        self.count += n
        self._render()

    def _render(self):
        if self.total:
            width = 28
            filled = int(width * self.count / self.total)
            bar = "#" * filled + "-" * (width - filled)
            pct = (self.count / self.total) * 100
            msg = f"{self.desc} [{bar}] {self.count}/{self.total} {pct:5.1f}%"
        else:
            msg = f"{self.desc} {self.count}"
        pad = max(0, self._last_len - len(msg))
        print("\r" + msg + (" " * pad), end="", flush=True)
        self._last_len = len(msg)

    def close(self):
        print()


def _iter_with_progress(items, total=None, desc=""):
    if tqdm is not None:
        for item in tqdm(items, total=total, desc=desc):
            yield item
    else:
        progress = _SimpleProgress(total=total, desc=desc)
        for item in items:
            yield item
            progress.update(1)
        progress.close()


def _progress_write(message):
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def load_tickers_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    candidates = ["symbol", "ticker", "Symbol", "Ticker", "SYMBOL", "TICKER"]
    symbol_col = next((c for c in candidates if c in df.columns), df.columns[0])

    tickers = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.US$", "", regex=True)
    )
    tickers = [t for t in tickers if t and t != "NAN"]
    return sorted(set(tickers))


def load_company_tickers(sec_client, cache_path=None):
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.values())

    data = sec_client.get_json("https://www.sec.gov/files/company_tickers.json")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return list(data.values())


def _extract_recent_filings(filings_json):
    if not filings_json:
        return {}
    filings = filings_json.get("filings")
    if isinstance(filings, dict) and filings.get("recent"):
        return filings.get("recent", {})
    if filings_json.get("recent"):
        return filings_json.get("recent", {})
    if isinstance(filings, dict):
        return filings
    return {}


def _filings_to_rows(recent):
    if not isinstance(recent, dict):
        return recent or []
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_times = recent.get("acceptanceDateTime", [])
    rows = []
    for i in range(len(accessions)):
        rows.append({
            "accessionNumber": accessions[i],
            "form": forms[i] if i < len(forms) else "",
            "filingDate": filing_dates[i] if i < len(filing_dates) else "",
            "acceptanceDateTime": acceptance_times[i] if i < len(acceptance_times) else "",
        })
    return rows


def iter_filings_for_cik(sec_client, cik, forms=("8-K",)):
    cik_padded = str(cik).zfill(10)
    root_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    root = sec_client.get_json(root_url)

    seen = set()

    recent = _extract_recent_filings(root)
    for row in _filings_to_rows(recent):
        if row.get("form") in forms:
            accession = row.get("accessionNumber", "")
            if accession and accession not in seen:
                seen.add(accession)
                yield row

    for file_info in (root.get("filings", {}).get("files") or []):
        name = file_info.get("name")
        if not name:
            continue
        file_url = f"https://data.sec.gov/submissions/{name}"
        try:
            data = sec_client.get_json(file_url)
        except requests.RequestException:
            continue
        recent = _extract_recent_filings(data)
        for row in _filings_to_rows(recent):
            if row.get("form") in forms:
                accession = row.get("accessionNumber", "")
                if accession and accession not in seen:
                    seen.add(accession)
                    yield row


def init_metadata_csv(csv_path):
    csv_path = Path(csv_path)
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cik", "ticker", "form", "filingDate", "acceptanceDateTime", "accessionNumber"],
        )
        writer.writeheader()


def append_metadata_rows(csv_path, rows):
    if not rows:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cik", "ticker", "form", "filingDate", "acceptanceDateTime", "accessionNumber"],
        )
        writer.writerows(rows)


def collect_filing_metadata(out_root="data/raw_data", forms=("8-K",), min_delay=0.2, limit_tickers=None, tickers_filter=None):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    cache_path = Path(out_root) / "_meta" / "company_tickers.json"
    tickers = load_company_tickers(sec_client, cache_path=cache_path)
    tickers = sorted(tickers, key=lambda x: x.get("ticker", ""))

    if tickers_filter:
        filter_set = {t.upper() for t in tickers_filter}
        tickers = [t for t in tickers if t.get("ticker", "").upper() in filter_set]

    if limit_tickers:
        tickers = tickers[: int(limit_tickers)]

    metadata_path = Path(out_root) / "filing_metadata.csv"
    init_metadata_csv(metadata_path)

    _progress_write(
        f"Starting filing metadata collection for {len(tickers)} tickers. "
        f"Output: {metadata_path} | Forms: {','.join(forms)} | min_delay={min_delay}s"
    )

    total_tickers = len(tickers)
    for idx, info in enumerate(_iter_with_progress(tickers, total=total_tickers, desc="Tickers"), 1):
        ticker = info.get("ticker", "")
        cik = info.get("cik_str", "")
        if not ticker or not cik:
            continue
        try:
            rows = []
            for filing in iter_filings_for_cik(sec_client, cik, forms=forms):
                rows.append({
                    "cik": str(cik).zfill(10),
                    "ticker": ticker.upper(),
                    "form": filing.get("form", ""),
                    "filingDate": filing.get("filingDate", ""),
                    "acceptanceDateTime": filing.get("acceptanceDateTime", ""),
                    "accessionNumber": filing.get("accessionNumber", ""),
                })
            append_metadata_rows(metadata_path, rows)
            if rows:
                _progress_write(f"[{idx}/{total_tickers}] {ticker}: rows {len(rows)}")
            else:
                _progress_write(
                    f"[{idx}/{total_tickers}] {ticker}: no filings for forms={','.join(forms)}"
                )
        except requests.RequestException as exc:
            _progress_write(
                f"[{idx}/{total_tickers}] {ticker}: failed - {exc} | forms={','.join(forms)}"
            )


if __name__ == "__main__":
    excel_path = "D:\\AQUMON\\us_symbol_list.xlsx"
    out_root = r"D:\AQUMON\data\raw_data"
    forms = ("4","6-K","8-K","10-K", "10-Q", "13F", "20-F", "40-F", "S-4",)
    # forms = ("13F",)
    min_delay = 1

    tickers_filter = load_tickers_from_excel(excel_path)
    _progress_write(f"Loaded {len(tickers_filter)} tickers from {excel_path}")

    collect_filing_metadata(
        out_root=out_root,
        forms=forms,
        min_delay=min_delay,
        tickers_filter=tickers_filter,
    )
