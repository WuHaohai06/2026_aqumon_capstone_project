# AQUMON Filing Research

This repository contains research scripts and notebooks for downloading SEC filings, extracting structured Form 4 data, scoring insider-trading sentiment, and running downstream comparison and event-study analysis.

## What Is In The Repo

- `scripts/download/`: SEC metadata and raw filing download helpers
- `scripts/datasets/`: golden dataset extraction
- `scripts/extraction/`: Form 4 structured extraction via Dify workflows
- `scripts/analysis/`: sentiment scoring, result comparison, and event-study helpers
- `prompts/`: prompt templates used by the extraction and sentiment pipelines
- `data/`: local raw data, results, and derived artifacts

## uv Workflow

This repository is set up as a non-package uv project.

### First-time setup

```bash
uv sync
```

This creates a local `.venv` and installs the default dependency groups, including notebook tooling.

### Common commands

```bash
uv run python scripts/analysis/build_sentiment_timeseries.py --help
uv run python scripts/download/download_raw_data_xml.py --help
uv run jupyter lab
```

### Use a minimal environment

If you only want the core script dependencies and do not need notebooks:

```bash
uv sync --no-default-groups
```

## Notes

- `.vscode/settings.json` points VS Code to `.venv`, so the workspace should prefer the uv environment automatically.
- `data/` is ignored by git because it contains local downloads and experiment outputs.
- Some Dify API settings are still hardcoded in the current research scripts. If this repo becomes a longer-lived shared project, those values should move to environment variables or config files.

## More Docs

- [scripts/README.md](scripts/README.md)
- [README_sentiment.md](README_sentiment.md)
- [README_compare_results.md](README_compare_results.md)
- [README_EVENT-STUDY-TOOLKIT.md](README_EVENT-STUDY-TOOLKIT.md)
