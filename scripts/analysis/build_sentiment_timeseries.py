from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_SOURCE = DATA_DIR / "results" / "4" / "4_golden_structured_input_gemi_batch"
DEFAULT_METADATA_PATH = DATA_DIR / "raw_data" / "filing_metadata.csv"
DEFAULT_CLOSE_ADJ_PATH = DATA_DIR / "close_adj.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "results" / "4" / "aligned_sentiment_timeseries"
DEFAULT_TIMEZONE = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a daily sentiment time series aligned to the trading calendar "
            "from close_adj.csv using SEC filing metadata and sentiment results."
        ),
    )
    parser.add_argument(
        "--results-source",
        default=str(DEFAULT_RESULTS_SOURCE),
        help=(
            "Sentiment results directory/file. Supports a results directory, "
            "sentiment_results.parquet, or sentiment_results.jsonl."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Path to filing_metadata.csv.",
    )
    parser.add_argument(
        "--close-adj-path",
        default=str(DEFAULT_CLOSE_ADJ_PATH),
        help="Path to close_adj.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write aligned outputs into.",
    )
    parser.add_argument(
        "--form",
        default="4",
        help="Only keep rows from filing_metadata matching this form.",
    )
    parser.add_argument(
        "--align-on",
        choices=("acceptance_datetime", "filing_date"),
        default="acceptance_datetime",
        help=(
            "Which metadata time field to anchor on before mapping to a trading day."
        ),
    )
    parser.add_argument(
        "--availability-rule",
        choices=("next_trading_day", "same_day_before_close"),
        default="next_trading_day",
        help=(
            "How to translate the event timestamp into a trading date. "
            "'next_trading_day' is conservative and avoids lookahead bias."
        ),
    )
    parser.add_argument(
        "--market-close-hour",
        type=int,
        default=16,
        help="Used only for same_day_before_close. Default assumes 16:00 local market time.",
    )
    parser.add_argument(
        "--market-timezone",
        default=DEFAULT_TIMEZONE,
        help="Timezone used when interpreting acceptanceDateTime.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help=(
            "Value used for missing daily sentiment cells in the aligned wide outputs. "
            "Use NaN by passing a non-finite value such as 'nan'."
        ),
    )
    return parser.parse_args()


def resolve_results_source(path: str | Path) -> Path:
    path = Path(path)

    if path.is_dir():
        candidates = [
            path / "sentiment_results.parquet",
            path / "sentiment_results.jsonl",
        ]
    elif path.suffix == ".parquet":
        candidates = [path, path.with_suffix(".jsonl")]
    elif path.suffix == ".jsonl":
        candidates = [path, path.with_suffix(".parquet")]
    else:
        candidates = [
            path / "sentiment_results.parquet",
            path / "sentiment_results.jsonl",
            path.with_suffix(".parquet"),
            path.with_suffix(".jsonl"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve results source from: {path}")


def confidence_to_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_results_table(path: str | Path) -> pd.DataFrame:
    source = resolve_results_source(path)

    if source.suffix == ".parquet":
        try:
            df = pd.read_parquet(source)
        except Exception:
            fallback_jsonl = source.with_suffix(".jsonl")
            if not fallback_jsonl.exists():
                raise
            source = fallback_jsonl
            df = pd.DataFrame(read_jsonl_records(source))
    elif source.suffix == ".jsonl":
        df = pd.DataFrame(read_jsonl_records(source))
    else:
        raise ValueError(f"Unsupported source type: {source}")

    required_columns = {"id", "sentiment_score", "confidence"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {source}")

    standardized = df.copy()
    standardized["id"] = standardized["id"].astype(str)
    standardized["sentiment_score"] = pd.to_numeric(
        standardized["sentiment_score"], errors="coerce"
    )
    standardized["confidence_pct"] = standardized["confidence"].map(confidence_to_float)
    standardized["source_path"] = str(source)
    return standardized


def prepare_output_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        if prepared[column].dtype != "object":
            continue
        prepared[column] = prepared[column].map(
            lambda value: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return prepared


def load_metadata(path: str | Path, form: str) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype=str)
    metadata = metadata.copy()
    metadata["form"] = metadata["form"].astype(str)
    metadata = metadata.loc[metadata["form"] == str(form)].copy()
    metadata = metadata.drop_duplicates(subset=["accessionNumber"], keep="first")
    metadata["id"] = metadata["accessionNumber"].astype(str)
    metadata["ticker"] = metadata["ticker"].where(metadata["ticker"].notna(), "").astype(str).str.upper()
    metadata["price_ticker"] = np.where(
        metadata["ticker"] != "",
        metadata["ticker"] + ".US",
        np.nan,
    )
    metadata["filing_date"] = pd.to_datetime(
        metadata["filingDate"], errors="coerce"
    ).dt.normalize()
    metadata["acceptance_datetime_utc"] = pd.to_datetime(
        metadata["acceptanceDateTime"], errors="coerce", utc=True
    )
    return metadata


def load_close_adj(path: str | Path) -> pd.DataFrame:
    close_adj = pd.read_csv(path)
    date_col = close_adj.columns[0]
    close_adj = close_adj.rename(columns={date_col: "date"})
    close_adj["date"] = pd.to_datetime(close_adj["date"], errors="coerce").dt.normalize()
    close_adj = close_adj.dropna(subset=["date"]).sort_values("date").set_index("date")
    close_adj.index.name = "date"
    close_adj.columns = close_adj.columns.astype(str)
    return close_adj


def map_to_next_trading_day(
    anchor_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp:
    if pd.isna(anchor_date):
        return pd.NaT
    pos = trading_dates.searchsorted(anchor_date.normalize(), side="right")
    if pos >= len(trading_dates):
        return pd.NaT
    return pd.Timestamp(trading_dates[pos]).normalize()


def map_to_same_or_next_trading_day(
    anchor_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp:
    if pd.isna(anchor_date):
        return pd.NaT
    pos = trading_dates.searchsorted(anchor_date.normalize(), side="left")
    if pos >= len(trading_dates):
        return pd.NaT
    return pd.Timestamp(trading_dates[pos]).normalize()


def determine_effective_trade_date(
    row: pd.Series,
    *,
    trading_dates: pd.DatetimeIndex,
    align_on: str,
    availability_rule: str,
    market_close_hour: int,
    market_timezone: str,
) -> pd.Timestamp:
    if align_on == "filing_date":
        anchor_date = row["filing_date"]
        if availability_rule == "next_trading_day":
            return map_to_next_trading_day(anchor_date, trading_dates)
        return map_to_same_or_next_trading_day(anchor_date, trading_dates)

    acceptance_ts = row["acceptance_datetime_utc"]
    if pd.isna(acceptance_ts):
        fallback_date = row["filing_date"]
        if availability_rule == "next_trading_day":
            return map_to_next_trading_day(fallback_date, trading_dates)
        return map_to_same_or_next_trading_day(fallback_date, trading_dates)

    local_ts = acceptance_ts.tz_convert(market_timezone)
    local_date = pd.Timestamp(local_ts.date())

    if availability_rule == "next_trading_day":
        return map_to_next_trading_day(local_date, trading_dates)

    if local_date in trading_dates and local_ts.hour < market_close_hour:
        return local_date.normalize()
    return map_to_next_trading_day(local_date, trading_dates)


def lookup_close(
    close_adj: pd.DataFrame,
    date: pd.Timestamp,
    price_ticker: str,
) -> float:
    if pd.isna(date):
        return np.nan
    if price_ticker not in close_adj.columns:
        return np.nan
    if date not in close_adj.index:
        return np.nan
    return close_adj.at[date, price_ticker]


def build_daily_aggregates(events: pd.DataFrame) -> pd.DataFrame:
    valid_events = events.dropna(subset=["effective_trade_date", "price_ticker"]).copy()
    valid_events = valid_events.loc[
        valid_events["sentiment_score"].notna() & valid_events["price_ticker"].notna()
    ].copy()

    if valid_events.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "price_ticker",
                "ticker",
                "sentiment_score_mean",
                "sentiment_score_sum",
                "sentiment_score_last",
                "confidence_pct_mean",
                "filing_count",
            ]
        )

    valid_events = valid_events.sort_values(
        ["effective_trade_date", "ticker", "acceptance_datetime_utc", "id"]
    )
    grouped = (
        valid_events.groupby(["effective_trade_date", "price_ticker", "ticker"], as_index=False)
        .agg(
            sentiment_score_mean=("sentiment_score", "mean"),
            sentiment_score_sum=("sentiment_score", "sum"),
            sentiment_score_last=("sentiment_score", "last"),
            confidence_pct_mean=("confidence_pct", "mean"),
            filing_count=("id", "count"),
        )
        .rename(columns={"effective_trade_date": "date"})
    )
    return grouped


def build_wide_metric_frames(
    daily: pd.DataFrame,
    close_adj: pd.DataFrame,
    metrics: Iterable[str],
    fill_value: float,
) -> dict[str, pd.DataFrame]:
    wide_frames: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        pivot = daily.pivot(index="date", columns="price_ticker", values=metric)
        pivot = pivot.reindex(index=close_adj.index, columns=close_adj.columns)
        if np.isfinite(fill_value):
            pivot = pivot.fillna(fill_value)
        wide_frames[metric] = pivot
    return wide_frames


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentiments = read_results_table(args.results_source)
    metadata = load_metadata(args.metadata_path, form=args.form)
    close_adj = load_close_adj(args.close_adj_path)
    trading_dates = pd.DatetimeIndex(close_adj.index.unique()).sort_values()

    events = sentiments.merge(
        metadata[
            [
                "id",
                "ticker",
                "price_ticker",
                "form",
                "filingDate",
                "filing_date",
                "acceptanceDateTime",
                "acceptance_datetime_utc",
            ]
        ],
        on="id",
        how="left",
    )
    events["acceptance_datetime_local"] = events["acceptance_datetime_utc"].dt.tz_convert(
        args.market_timezone
    )
    events["effective_trade_date"] = events.apply(
        determine_effective_trade_date,
        axis=1,
        trading_dates=trading_dates,
        align_on=args.align_on,
        availability_rule=args.availability_rule,
        market_close_hour=args.market_close_hour,
        market_timezone=args.market_timezone,
    )
    events["close_adj_on_effective_date"] = [
        lookup_close(close_adj, row.effective_trade_date, row.price_ticker)
        for row in events.itertuples()
    ]
    events["has_metadata_match"] = events["ticker"].notna()
    events["has_close_adj_match"] = events["close_adj_on_effective_date"].notna()

    daily = build_daily_aggregates(events)
    metrics = [
        "sentiment_score_mean",
        "sentiment_score_sum",
        "sentiment_score_last",
        "confidence_pct_mean",
        "filing_count",
    ]
    wide_frames = build_wide_metric_frames(
        daily=daily,
        close_adj=close_adj,
        metrics=metrics,
        fill_value=args.fill_value,
    )

    events_output = output_dir / "event_level_sentiment.parquet"
    daily_output = output_dir / "daily_sentiment_events_long.parquet"
    config_output = output_dir / "alignment_config.json"
    prepare_output_table(events).to_parquet(events_output, index=False)
    prepare_output_table(daily).to_parquet(daily_output, index=False)

    for metric, frame in wide_frames.items():
        frame.to_parquet(output_dir / f"{metric}_aligned_to_close_adj.parquet")

    config_output.write_text(
        json.dumps(
            {
                "results_source": str(resolve_results_source(args.results_source)),
                "metadata_path": str(Path(args.metadata_path).resolve()),
                "close_adj_path": str(Path(args.close_adj_path).resolve()),
                "output_dir": str(output_dir.resolve()),
                "form": args.form,
                "align_on": args.align_on,
                "availability_rule": args.availability_rule,
                "market_close_hour": args.market_close_hour,
                "market_timezone": args.market_timezone,
                "fill_value": args.fill_value,
                "event_rows": int(len(events)),
                "event_rows_with_metadata": int(events["has_metadata_match"].sum()),
                "event_rows_with_price_match": int(events["has_close_adj_match"].sum()),
                "daily_event_rows": int(len(daily)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote event table: {events_output}")
    print(f"Wrote sparse daily event table: {daily_output}")
    for metric in metrics:
        print(f"Wrote aligned daily matrix: {output_dir / f'{metric}_aligned_to_close_adj.parquet'}")
    print(f"Wrote config summary: {config_output}")


if __name__ == "__main__":
    main()
