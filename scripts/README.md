# Scripts Layout

The Python entrypoints are grouped by responsibility under `scripts/`:

- `scripts/download/`: SEC metadata and raw filing download
- `scripts/datasets/`: golden dataset extraction
- `scripts/extraction/`: Form 4 structured extraction
- `scripts/analysis/`: sentiment scoring, time-series alignment, comparison, and event-study helpers
- `scripts/utils/`: shared utility helpers

## Common uv Commands

Run these from the repository root:

```bash
uv run python scripts/download/download_raw_data_xml.py --help
uv run python scripts/datasets/extract_golden_dataset.py --help
uv run python scripts/extraction/form4_structured_extraction.py --help
uv run python scripts/analysis/sentiment.py --help
uv run python scripts/analysis/sentiment_structured.py --help
uv run python scripts/analysis/build_sentiment_timeseries.py --help
```

`scripts/download/download_filing_metadata.py` currently runs from hardcoded defaults in its `__main__` block and does not expose an argparse `--help` entry yet.

To use the comparison helpers in a notebook:

```python
from scripts.analysis import compare_results as cr
```
