"""
Sample Script for Downloading Report Content from SEC EDGAR
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import re
import json
import time
import multiprocessing as mp
import shutil
from pathlib import Path
from urllib.parse import urljoin, urlparse
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_latest_8k(ticker):
    headers = {'User-Agent': 'given.family@magnumwm.com'}

    # Get CIK from ticker
    ticker_cik = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    ).json()
    cik = str([v['cik_str'] for k, v in ticker_cik.items() if v['ticker'] == ticker.upper()][0])

    # Get recent filings
    submissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
        headers=headers
    ).json()

    # Find latest 8-K
    recent = submissions['filings']['recent']
    df = pd.DataFrame(recent)
    latest_8k = df[df['form'] == '8-K'].iloc[0]

    accession = latest_8k['accessionNumber'].replace('-', '')
    primary_doc = latest_8k['primaryDocument']

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    return url


def download_text(filing_url, sec_client=None):
    headers = {'User-Agent': 'given.family@magnumwm.com'}
    if sec_client is not None:
        response = sec_client.get(filing_url)
    else:
        response = requests.get(filing_url, headers=headers)
        response.raise_for_status()

    # removes HTML tables, scripts, styles
    soup = BeautifulSoup(response.text, 'html.parser')

    # Kill scripts, styles, and navigation junk
    for script in soup(["script", "style", "header", "footer", "nav"]):
        script.decompose()

    text = soup.get_text(separator='\n')

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def download_clean_text_file(sec_client, filing_url, out_dir, filename, skip_existing=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / ".done"
    dest = out_dir / filename

    if skip_existing and done_marker.exists():
        return {
            "downloaded": 0,
            "skipped": 0,
            "total": 1,
            "done": True,
        }

    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        done_marker.write_text("ok", encoding="utf-8")
        return {
            "downloaded": 0,
            "skipped": 1,
            "total": 1,
            "done": True,
        }

    text = download_text(filing_url, sec_client=sec_client)
    dest.write_text(text, encoding="utf-8", errors="ignore")
    done_marker.write_text("ok", encoding="utf-8")
    return {
        "downloaded": 1,
        "skipped": 0,
        "total": 1,
        "done": True,
    }


def _to_accession_dash(accession_nodash):
    if re.fullmatch(r"\d{18}", accession_nodash):
        return f"{accession_nodash[:10]}-{accession_nodash[10:12]}-{accession_nodash[12:]}"
    return accession_nodash


def get_filing_dir_info(filing_url):
    parsed = urlparse(filing_url)
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2:
        raise ValueError(f"Unexpected filing URL format: {filing_url}")

    accession_nodash = None
    accession_index = None
    for i in range(len(segments) - 1, -1, -1):
        if re.fullmatch(r"\d{18}", segments[i]):
            accession_nodash = segments[i]
            accession_index = i
            break
    if not accession_nodash:
        raise ValueError(f"Unexpected filing URL format: {filing_url}")

    accession_dash = _to_accession_dash(accession_nodash)
    base_dir = f"{parsed.scheme}://{parsed.netloc}/" + "/".join(segments[:accession_index + 1]) + "/"
    return base_dir, accession_nodash, accession_dash


def list_txt_files_in_filing(filing_url):
    headers = {'User-Agent': 'given.family@magnumwm.com'}
    base_dir, _, accession_dash = get_filing_dir_info(filing_url)

    # Prefer index.json if available
    index_json_url = urljoin(base_dir, "index.json")
    try:
        response = requests.get(index_json_url, headers=headers, timeout=30)
        if response.ok:
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            txt_urls = [
                urljoin(base_dir, item.get("name", ""))
                for item in items
                if item.get("name", "").lower().endswith(".txt")
            ]
            if txt_urls:
                return txt_urls
    except (ValueError, requests.RequestException):
        pass

    # Fallback to the filing index HTML
    index_html_url = urljoin(base_dir, f"{accession_dash}-index.html")
    response = requests.get(index_html_url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    txt_urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".txt"):
            txt_urls.append(urljoin(index_html_url, href))

    return txt_urls


def download_all_txt_files(filing_url, out_dir):
    headers = {'User-Agent': 'given.family@magnumwm.com'}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_urls = list_txt_files_in_filing(filing_url)
    downloaded = []
    seen = set()
    for url in txt_urls:
        if url in seen:
            continue
        seen.add(url)
        filename = url.split("/")[-1]
        dest = out_dir / filename
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        downloaded.append(str(dest))

    return downloaded


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


def _normalize_download_mode(mode):
    if not mode:
        return "clean"
    mode = str(mode).strip().lower()
    if mode in ("clean", "raw"):
        return mode
    raise ValueError(f"Unsupported download_mode: {mode}")


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


def prune_downloads_folder(out_root, allowed_tickers, forms=None):
    out_root = Path(out_root)
    if not out_root.exists():
        return

    allowed = {t.upper() for t in allowed_tickers}
    allowed.add("_META")
    form_dirs = {_form_to_dirname(f).upper() for f in (forms or [])}

    for entry in out_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.upper()
        if name in allowed:
            continue
        if name in form_dirs:
            for sym_dir in entry.iterdir():
                if not sym_dir.is_dir():
                    continue
                sym_name = sym_dir.name.upper()
                if sym_name not in allowed:
                    _progress_write(f"Removing extra folder: {sym_dir}")
                    shutil.rmtree(sym_dir, ignore_errors=True)
            continue
        _progress_write(f"Removing extra folder: {entry}")
        shutil.rmtree(entry, ignore_errors=True)


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
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_times = recent.get("acceptanceDateTime", [])
    rows = []
    for i in range(len(accessions)):
        rows.append({
            "accessionNumber": accessions[i],
            "form": forms[i] if i < len(forms) else "",
            "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
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


def build_filing_url(cik, accession_nodash, primary_document=None):
    doc = primary_document or "index.html"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"


def list_txt_files_in_filing_sec(sec_client, filing_url):
    base_dir, _, accession_dash = get_filing_dir_info(filing_url)

    index_json_url = urljoin(base_dir, "index.json")
    try:
        response = sec_client.get(index_json_url, timeout=30)
        data = response.json()
        items = data.get("directory", {}).get("item", [])
        txt_urls = [
            urljoin(base_dir, item.get("name", ""))
            for item in items
            if item.get("name", "").lower().endswith(".txt")
        ]
        if txt_urls:
            return txt_urls
    except (ValueError, requests.RequestException, RuntimeError):
        pass

    index_html_url = urljoin(base_dir, f"{accession_dash}-index.html")
    try:
        response = sec_client.get(index_html_url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
        txt_urls = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".txt"):
                txt_urls.append(urljoin(index_html_url, href))
        return txt_urls
    except (requests.RequestException, RuntimeError):
        return []


def download_all_txt_files_sec(sec_client, filing_url, out_dir, skip_existing=True):
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
        p.name
        for p in out_dir.glob("*.txt")
        if p.is_file() and p.stat().st_size > 0
    }

    txt_urls = list_txt_files_in_filing_sec(sec_client, filing_url)
    if skip_existing and txt_urls and existing_files:
        expected = {url.split("/")[-1] for url in txt_urls}
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
    for url in txt_urls:
        if url in seen:
            continue
        seen.add(url)
        filename = url.split("/")[-1]
        dest = out_dir / filename
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        response = sec_client.get(url, timeout=60)
        with open(dest, "wb") as f:
            f.write(response.content)
        downloaded.append(str(dest))

    result = {
        "downloaded": len(downloaded),
        "skipped": skipped,
        "total": len(txt_urls),
        "done": False,
    }
    if result["total"] > 0:
        done_marker.write_text("ok", encoding="utf-8")
        result["done"] = True
    return result


def download_all_txt_for_ticker(sec_client, ticker, cik, out_root, forms=("8-K",), download_mode="clean"):
    filings_total = 0
    filings_with_txt = 0
    filings_done = 0
    txt_total = 0
    txt_downloaded = 0
    txt_skipped = 0
    download_mode = _normalize_download_mode(download_mode)
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
            if download_mode == "raw":
                result = download_all_txt_files_sec(sec_client, filing_url, out_dir, skip_existing=True)
            else:
                filename = f"{accession_dash}.txt"
                result = download_clean_text_file(
                    sec_client,
                    filing_url,
                    out_dir,
                    filename=filename,
                    skip_existing=True,
                )
            if result.get("done"):
                filings_done += 1
            if result["total"] > 0:
                filings_with_txt += 1
            txt_total += result["total"]
            txt_downloaded += result["downloaded"]
            txt_skipped += result["skipped"]
        except requests.RequestException:
            continue
    return {
        "filings": filings_total,
        "filings_with_txt": filings_with_txt,
        "filings_done": filings_done,
        "txt_total": txt_total,
        "txt_downloaded": txt_downloaded,
        "txt_skipped": txt_skipped,
    }


def download_all_txt_for_all_tickers(out_root="downloads", forms=("8-K",), min_delay=0.2, limit_tickers=None, tickers_filter=None, download_mode="clean"):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    cache_path = Path(out_root) / "_meta" / "company_tickers.json"
    tickers = load_company_tickers(sec_client, cache_path=cache_path)
    tickers = sorted(tickers, key=lambda x: x.get("ticker", ""))
    download_mode = _normalize_download_mode(download_mode)

    if tickers_filter:
        filter_set = {t.upper() for t in tickers_filter}
        tickers = [t for t in tickers if t.get("ticker", "").upper() in filter_set]

    if limit_tickers:
        tickers = tickers[: int(limit_tickers)]

    _progress_write(
        f"Starting txt download for {len(tickers)} tickers. "
        f"Output: {out_root} | Forms: {','.join(forms)} | mode={download_mode} | min_delay={min_delay}s"
    )

    total_tickers = len(tickers)
    for idx, info in enumerate(_iter_with_progress(tickers, total=total_tickers, desc="Tickers"), 1):
        ticker = info.get("ticker", "")
        cik = info.get("cik_str", "")
        if not ticker or not cik:
            continue
        try:
            stats = download_all_txt_for_ticker(
                sec_client,
                ticker=ticker,
                cik=cik,
                out_root=out_root,
                forms=forms,
                download_mode=download_mode,
            )
            _progress_write(
                f"[{idx}/{total_tickers}] {ticker}: "
                f"filings {stats['filings']} (done {stats['filings_done']}, with txt {stats['filings_with_txt']}), "
                f"txt total {stats['txt_total']}, downloaded {stats['txt_downloaded']}, "
                f"skipped {stats['txt_skipped']}"
            )
        except requests.RequestException as exc:
            _progress_write(f"[{idx}/{total_tickers}] {ticker}: failed - {exc}")


def _download_ticker_worker(args):
    ticker, cik, out_root, forms, min_delay, download_mode = args
    try:
        sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
        stats = download_all_txt_for_ticker(
            sec_client=sec_client,
            ticker=ticker,
            cik=cik,
            out_root=out_root,
            forms=forms,
            download_mode=download_mode,
        )
        return ticker, stats, None
    except Exception as exc:
        return ticker, None, str(exc)


def download_all_txt_for_all_tickers_mp(
    out_root="downloads",
    forms=("8-K",),
    min_delay=0.2,
    limit_tickers=None,
    tickers_filter=None,
    num_workers=4,
    download_mode="clean",
):
    sec_client = SecClient(user_agent="given.family@magnumwm.com", min_delay=min_delay)
    cache_path = Path(out_root) / "_meta" / "company_tickers.json"
    tickers = load_company_tickers(sec_client, cache_path=cache_path)
    tickers = sorted(tickers, key=lambda x: x.get("ticker", ""))
    download_mode = _normalize_download_mode(download_mode)

    if tickers_filter:
        filter_set = {t.upper() for t in tickers_filter}
        tickers = [t for t in tickers if t.get("ticker", "").upper() in filter_set]

    if limit_tickers:
        tickers = tickers[: int(limit_tickers)]

    tasks = []
    for info in tickers:
        ticker = info.get("ticker", "")
        cik = info.get("cik_str", "")
        if ticker and cik:
            tasks.append((ticker, cik, out_root, forms, min_delay, download_mode))

    _progress_write(
        f"Starting txt download with {num_workers} workers for {len(tasks)} tickers. "
        f"Output: {out_root} | Forms: {','.join(forms)} | mode={download_mode} | min_delay={min_delay}s"
    )

    total_tickers = len(tasks)
    with mp.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_download_ticker_worker, tasks)
        for idx, result in enumerate(_iter_with_progress(iterator, total=total_tickers, desc="Tickers"), 1):
            ticker, stats, err = result
            if err:
                _progress_write(f"[{idx}/{total_tickers}] {ticker}: failed - {err}")
                continue
            _progress_write(
                f"[{idx}/{total_tickers}] {ticker}: "
                f"filings {stats['filings']} (done {stats['filings_done']}, with txt {stats['filings_with_txt']}), "
                f"txt total {stats['txt_total']}, downloaded {stats['txt_downloaded']}, "
                f"skipped {stats['txt_skipped']}"
            )
    
# ==================== TEST ====================
if __name__ == "__main__":

    # Usage
    excel_path = REPO_ROOT / "us_symbol_list.xlsx"
    out_root = REPO_ROOT / "data" / "raw_data"
    # forms = ("4","6-K","8-K","10-K", "10-Q", "13F", "20-F", "40-F", "S-4",)
    forms = ("4","6-K", "13F", "20-F", "40-F", "S-4",)
    # forms = ("6-K", "8-K",)
    min_delay = 1
    download_mode = "clean"  # "clean" uses download_text(), "raw" downloads all .txt attachments

    tickers_filter = load_tickers_from_excel(excel_path)
    _progress_write(f"Loaded {len(tickers_filter)} tickers from {excel_path}")
    prune_enabled = False
    if prune_enabled:
        prune_downloads_folder(out_root, tickers_filter, forms=forms)

    # Small batch test: limit to the first 5 tickers in the SEC list
    # download_all_txt_for_all_tickers(out_root=out_root, forms=forms, min_delay=min_delay, limit_tickers=5, tickers_filter=tickers_filter)
    # Multiprocess example:
    download_all_txt_for_all_tickers_mp(
        out_root=out_root,
        forms=forms,
        min_delay=min_delay,
        num_workers=16,
        tickers_filter=tickers_filter,
        download_mode=download_mode,
    )
