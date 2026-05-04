"""
Download raw XML attachments from SEC EDGAR filing directories.

Supports two modes:
1. Full ticker universe traversal via SEC submissions.
2. Golden dataset subset traversal using existing directories under extracted_raw_data.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _form_to_dirname(form):
    if not form:
        return "UNKNOWN"
    return str(form).strip().upper().replace("/", "-")


def _normalize_forms(forms):
    if not forms:
        return None
    return tuple(_form_to_dirname(form) for form in forms)


def _accession_to_cik(accession_dash):
    digits = re.sub(r"\D", "", str(accession_dash or ""))
    if len(digits) < 10:
        raise ValueError(f"Unexpected accession format: {accession_dash}")
    return str(int(digits[:10]))


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
            with open(cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return list(data.values())

    data = sec_client.get_json("https://www.sec.gov/files/company_tickers.json")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
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
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_times = recent.get("acceptanceDateTime", [])
    rows = []
    for index in range(len(accessions)):
        rows.append(
            {
                "accessionNumber": accessions[index],
                "form": forms[index] if index < len(forms) else "",
                "primaryDocument": primary_docs[index] if index < len(primary_docs) else "",
                "filingDate": filing_dates[index] if index < len(filing_dates) else "",
                "acceptanceDateTime": acceptance_times[index] if index < len(acceptance_times) else "",
            }
        )
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
                row["accession_nodash"] = accession.replace("-", "")
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
                    row["accession_nodash"] = accession.replace("-", "")
                    yield row


def _to_accession_dash(accession_nodash):
    if re.fullmatch(r"\d{18}", accession_nodash):
        return f"{accession_nodash[:10]}-{accession_nodash[10:12]}-{accession_nodash[12:]}"
    return accession_nodash


def build_filing_url(cik, accession_nodash, primary_document=None):
    doc = primary_document or "index.html"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"


def get_filing_dir_info(filing_url):
    parsed = urlparse(filing_url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        raise ValueError(f"Unexpected filing URL format: {filing_url}")

    accession_nodash = None
    accession_index = None
    for index in range(len(segments) - 1, -1, -1):
        if re.fullmatch(r"\d{18}", segments[index]):
            accession_nodash = segments[index]
            accession_index = index
            break

    if not accession_nodash:
        raise ValueError(f"Unexpected filing URL format: {filing_url}")

    accession_dash = _to_accession_dash(accession_nodash)
    base_dir = f"{parsed.scheme}://{parsed.netloc}/" + "/".join(segments[:accession_index + 1]) + "/"
    return base_dir, accession_dash


def list_xml_files_in_filing_sec(sec_client, filing_url):
    base_dir, accession_dash = get_filing_dir_info(filing_url)

    index_json_url = urljoin(base_dir, "index.json")
    try:
        response = sec_client.get(index_json_url, timeout=30)
        data = response.json()
        items = data.get("directory", {}).get("item", [])
        xml_urls = [
            urljoin(base_dir, item.get("name", ""))
            for item in items
            if item.get("name", "").lower().endswith(".xml")
        ]
        if xml_urls:
            return xml_urls
    except (ValueError, requests.RequestException, RuntimeError):
        pass

    index_html_url = urljoin(base_dir, f"{accession_dash}-index.html")
    try:
        response = sec_client.get(index_html_url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        xml_urls = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".xml"):
                xml_urls.append(urljoin(index_html_url, href))
        return xml_urls
    except (requests.RequestException, RuntimeError):
        return []


def download_all_xml_files_sec(sec_client, filing_url, out_dir, skip_existing=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / ".done"
    if skip_existing and done_marker.exists():
        return {
            "downloaded": 0,
            "skipped": 0,
            "total": 0,
            "done": True,
        }

    existing_files = {
        path.name
        for path in out_dir.glob("*.xml")
        if path.is_file() and path.stat().st_size > 0
    }

    xml_urls = list_xml_files_in_filing_sec(sec_client, filing_url)
    if skip_existing and xml_urls and existing_files:
        expected = {url.split("/")[-1] for url in xml_urls}
        if expected.issubset(existing_files):
            done_marker.write_text("ok", encoding="utf-8")
            return {
                "downloaded": 0,
                "skipped": len(expected),
                "total": len(expected),
                "done": True,
            }

    downloaded = []
    skipped = 0
    seen = set()
    for url in xml_urls:
        if url in seen:
            continue
        seen.add(url)
        filename = url.split("/")[-1]
        dest = out_dir / filename
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        response = sec_client.get(url, timeout=60)
        with open(dest, "wb") as handle:
            handle.write(response.content)
        downloaded.append(str(dest))

    result = {
        "downloaded": len(downloaded),
        "skipped": skipped,
        "total": len(xml_urls),
        "done": False,
    }
    if result["total"] > 0:
        done_marker.write_text("ok", encoding="utf-8")
        result["done"] = True
    return result


def download_all_xml_for_ticker(sec_client, ticker, cik, out_root, forms=("8-K",)):
    filings_total = 0
    filings_with_xml = 0
    filings_done = 0
    xml_total = 0
    xml_downloaded = 0
    xml_skipped = 0

    for filing in iter_filings_for_cik(sec_client, cik, forms=forms):
        filings_total += 1
        accession_nodash = filing.get("accession_nodash") or filing.get("accessionNumber", "").replace("-", "")
        if not accession_nodash:
            continue
        accession_dash = _to_accession_dash(accession_nodash)
        form_dir = _form_to_dirname(filing.get("form"))
        out_dir = Path(out_root) / form_dir / ticker.upper() / accession_dash
        filing_url = build_filing_url(cik, accession_nodash, filing.get("primaryDocument"))
        try:
            result = download_all_xml_files_sec(sec_client, filing_url, out_dir, skip_existing=True)
            if result.get("done"):
                filings_done += 1
            if result["total"] > 0:
                filings_with_xml += 1
            xml_total += result["total"]
            xml_downloaded += result["downloaded"]
            xml_skipped += result["skipped"]
        except requests.RequestException:
            continue

    return {
        "filings": filings_total,
        "filings_with_xml": filings_with_xml,
        "filings_done": filings_done,
        "xml_total": xml_total,
        "xml_downloaded": xml_downloaded,
        "xml_skipped": xml_skipped,
    }


def download_all_xml_for_all_tickers(out_root="downloads", forms=("8-K",), min_delay=0.2, limit_tickers=None, tickers_filter=None):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    cache_path = Path(out_root) / "_meta" / "company_tickers.json"
    tickers = load_company_tickers(sec_client, cache_path=cache_path)
    tickers = sorted(tickers, key=lambda x: x.get("ticker", ""))

    if tickers_filter:
        filter_set = {ticker.upper() for ticker in tickers_filter}
        tickers = [ticker_info for ticker_info in tickers if ticker_info.get("ticker", "").upper() in filter_set]

    if limit_tickers:
        tickers = tickers[: int(limit_tickers)]

    _progress_write(
        f"Starting xml download for {len(tickers)} tickers. "
        f"Output: {out_root} | Forms: {','.join(forms)} | min_delay={min_delay}s"
    )

    total_tickers = len(tickers)
    for index, info in enumerate(_iter_with_progress(tickers, total=total_tickers, desc="Tickers"), 1):
        ticker = info.get("ticker", "")
        cik = info.get("cik_str", "")
        if not ticker or not cik:
            continue
        try:
            stats = download_all_xml_for_ticker(
                sec_client,
                ticker=ticker,
                cik=cik,
                out_root=out_root,
                forms=forms,
            )
            _progress_write(
                f"[{index}/{total_tickers}] {ticker}: "
                f"filings {stats['filings']} (done {stats['filings_done']}, with xml {stats['filings_with_xml']}), "
                f"xml total {stats['xml_total']}, downloaded {stats['xml_downloaded']}, "
                f"skipped {stats['xml_skipped']}"
            )
        except requests.RequestException as exc:
            _progress_write(f"[{index}/{total_tickers}] {ticker}: failed - {exc}")


def _download_ticker_worker(args):
    ticker, cik, out_root, forms, min_delay = args
    try:
        sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
        stats = download_all_xml_for_ticker(
            sec_client=sec_client,
            ticker=ticker,
            cik=cik,
            out_root=out_root,
            forms=forms,
        )
        return ticker, stats, None
    except Exception as exc:
        return ticker, None, str(exc)


def download_all_xml_for_all_tickers_mp(
    out_root="downloads",
    forms=("8-K",),
    min_delay=0.2,
    limit_tickers=None,
    tickers_filter=None,
    num_workers=4,
):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    cache_path = Path(out_root) / "_meta" / "company_tickers.json"
    tickers = load_company_tickers(sec_client, cache_path=cache_path)
    tickers = sorted(tickers, key=lambda x: x.get("ticker", ""))

    if tickers_filter:
        filter_set = {ticker.upper() for ticker in tickers_filter}
        tickers = [ticker_info for ticker_info in tickers if ticker_info.get("ticker", "").upper() in filter_set]

    if limit_tickers:
        tickers = tickers[: int(limit_tickers)]

    tasks = []
    for info in tickers:
        ticker = info.get("ticker", "")
        cik = info.get("cik_str", "")
        if ticker and cik:
            tasks.append((ticker, cik, out_root, forms, min_delay))

    _progress_write(
        f"Starting xml download with {num_workers} workers for {len(tasks)} tickers. "
        f"Output: {out_root} | Forms: {','.join(forms)} | min_delay={min_delay}s"
    )

    total_tickers = len(tasks)
    with mp.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_download_ticker_worker, tasks)
        for index, result in enumerate(_iter_with_progress(iterator, total=total_tickers, desc="Tickers"), 1):
            ticker, stats, err = result
            if err:
                _progress_write(f"[{index}/{total_tickers}] {ticker}: failed - {err}")
                continue
            _progress_write(
                f"[{index}/{total_tickers}] {ticker}: "
                f"filings {stats['filings']} (done {stats['filings_done']}, with xml {stats['filings_with_xml']}), "
                f"xml total {stats['xml_total']}, downloaded {stats['xml_downloaded']}, "
                f"skipped {stats['xml_skipped']}"
            )


def iter_dataset_filing_entries(source_root, forms=None):
    source_root = Path(source_root)
    allowed_forms = set(_normalize_forms(forms) or [])
    if not source_root.exists():
        return

    for form_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if allowed_forms and form_dir.name.upper() not in allowed_forms:
            continue
        for ticker_dir in sorted(path for path in form_dir.iterdir() if path.is_dir()):
            for accession_dir in sorted(path for path in ticker_dir.iterdir() if path.is_dir()):
                accession_dash = accession_dir.name
                accession_nodash = accession_dash.replace("-", "")
                if not re.fullmatch(r"\d{18}", accession_nodash):
                    continue
                yield {
                    "form": form_dir.name,
                    "ticker": ticker_dir.name,
                    "accession_dash": accession_dash,
                    "accession_nodash": accession_nodash,
                    "cik": _accession_to_cik(accession_dash),
                    "source_dir": accession_dir,
                }


def download_xml_for_dataset_entry(sec_client, entry, out_root):
    filing_url = build_filing_url(entry["cik"], entry["accession_nodash"], None)
    out_dir = entry["source_dir"] if out_root is None else Path(out_root) / entry["form"] / entry["ticker"] / entry["accession_dash"]
    result = download_all_xml_files_sec(sec_client, filing_url, out_dir, skip_existing=True)
    result["out_dir"] = str(out_dir)
    return result


def download_xml_for_dataset_dirs(source_root, out_root, forms=None, min_delay=0.2):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    entries = list(iter_dataset_filing_entries(source_root, forms=forms) or [])
    _progress_write(
        f"Starting golden-dataset xml download for {len(entries)} filings. "
        f"Source: {source_root} | Output: {out_root} | Forms: {','.join(forms) if forms else 'ALL'} | min_delay={min_delay}s"
    )

    summary = {
        "filings": len(entries),
        "filings_with_xml": 0,
        "filings_done": 0,
        "xml_total": 0,
        "xml_downloaded": 0,
        "xml_skipped": 0,
    }
    for index, entry in enumerate(_iter_with_progress(entries, total=len(entries), desc="Filings"), 1):
        try:
            result = download_xml_for_dataset_entry(sec_client, entry, out_root)
        except requests.RequestException as exc:
            _progress_write(
                f"[{index}/{len(entries)}] {entry['ticker']} {entry['accession_dash']}: failed - {exc}"
            )
            continue

        if result.get("done"):
            summary["filings_done"] += 1
        if result["total"] > 0:
            summary["filings_with_xml"] += 1
        summary["xml_total"] += result["total"]
        summary["xml_downloaded"] += result["downloaded"]
        summary["xml_skipped"] += result["skipped"]
        _progress_write(
            f"[{index}/{len(entries)}] {entry['ticker']} {entry['accession_dash']}: "
            f"xml total {result['total']}, downloaded {result['downloaded']}, skipped {result['skipped']}"
        )

    return summary


def _download_dataset_entry_worker(args):
    entry, out_root, min_delay = args
    try:
        sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
        result = download_xml_for_dataset_entry(sec_client, entry, out_root)
        return entry, result, None
    except Exception as exc:
        return entry, None, str(exc)


def download_xml_for_dataset_dirs_mp(source_root, out_root, forms=None, min_delay=0.2, num_workers=4):
    entries = list(iter_dataset_filing_entries(source_root, forms=forms) or [])
    _progress_write(
        f"Starting golden-dataset xml download with {num_workers} workers for {len(entries)} filings. "
        f"Source: {source_root} | Output: {out_root} | Forms: {','.join(forms) if forms else 'ALL'} | min_delay={min_delay}s"
    )

    tasks = [(entry, out_root, min_delay) for entry in entries]
    summary = {
        "filings": len(entries),
        "filings_with_xml": 0,
        "filings_done": 0,
        "xml_total": 0,
        "xml_downloaded": 0,
        "xml_skipped": 0,
    }
    with mp.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_download_dataset_entry_worker, tasks)
        for index, result in enumerate(_iter_with_progress(iterator, total=len(tasks), desc="Filings"), 1):
            entry, stats, err = result
            if err:
                _progress_write(
                    f"[{index}/{len(tasks)}] {entry['ticker']} {entry['accession_dash']}: failed - {err}"
                )
                continue

            if stats.get("done"):
                summary["filings_done"] += 1
            if stats["total"] > 0:
                summary["filings_with_xml"] += 1
            summary["xml_total"] += stats["total"]
            summary["xml_downloaded"] += stats["downloaded"]
            summary["xml_skipped"] += stats["skipped"]
            _progress_write(
                f"[{index}/{len(tasks)}] {entry['ticker']} {entry['accession_dash']}: "
                f"xml total {stats['total']}, downloaded {stats['downloaded']}, skipped {stats['skipped']}"
            )

    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Download SEC XML attachments.")
    parser.add_argument(
        "--mode",
        choices=("golden", "all-tickers"),
        default="golden",
        help="Download mode. 'golden' only downloads filings already selected into the golden dataset.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "data" / "golden_dataset_engine" / "extracted_raw_data",
        help="Source root for golden mode; existing filing directories under this path define the subset to download.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. In golden mode, omit this to download XML directly into existing filing folders under --source-root.",
    )
    parser.add_argument(
        "--forms",
        nargs="*",
        default=["4"],
        help="Form types to include. Default is only Form 4 for golden validation.",
    )
    parser.add_argument("--min-delay", type=float, default=1.0, help="Minimum delay between SEC requests per worker.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=REPO_ROOT / "us_symbol_list.xlsx",
        help="Ticker Excel file used only in all-tickers mode.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    forms = tuple(args.forms) if args.forms else None

    if args.mode == "golden":
        summary = download_xml_for_dataset_dirs_mp(
            source_root=args.source_root,
            out_root=args.out_root,
            forms=forms,
            min_delay=args.min_delay,
            num_workers=args.num_workers,
        )
        output_target = args.out_root or args.source_root
        print(
            f"[SUMMARY] filings={summary['filings']} with_xml={summary['filings_with_xml']} "
            f"downloaded={summary['xml_downloaded']} skipped={summary['xml_skipped']}"
        )
        print(f"[SUMMARY] output={output_target}")
        return 0

    out_root = args.out_root or (REPO_ROOT / "data" / "raw_data_xml")
    tickers_filter = load_tickers_from_excel(args.excel_path)
    _progress_write(f"Loaded {len(tickers_filter)} tickers from {args.excel_path}")
    download_all_xml_for_all_tickers_mp(
        out_root=out_root,
        forms=forms or ("4",),
        min_delay=args.min_delay,
        num_workers=args.num_workers,
        tickers_filter=tickers_filter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
