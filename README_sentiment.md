# Sentiment Pipelines

This repository contains two main sentiment scoring flows for Form 4 research:

- `scripts/analysis/sentiment.py`
  - Runs sentiment scoring directly from raw Form 4 text files
- `scripts/analysis/sentiment_structured.py`
  - Runs sentiment scoring from structured JSON produced by the extraction pipeline

## Environment

Install the project environment with uv:

```bash
uv sync
```

## Typical commands

Raw-text pipeline:

```bash
uv run python scripts/analysis/sentiment.py --help
```

Structured-input pipeline:

```bash
uv run python scripts/analysis/sentiment_structured.py --help
```

Time-series alignment after scoring:

```bash
uv run python scripts/analysis/build_sentiment_timeseries.py --help
```

## Outputs

Depending on the script and prompt, runs usually write:

- `sentiment_results.jsonl`
- `failed_transcripts.jsonl`
- `sentiment_results.parquet`
- `sentiment_pipeline.log`

The structured pipeline is typically the better starting point for downstream event studies because it produces more stable, easier-to-audit signal inputs.
