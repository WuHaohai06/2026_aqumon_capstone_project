# Compare Results

`scripts/analysis/compare_results.py` is a notebook-oriented helper module for comparing multiple sentiment result sets.

It is designed to answer questions such as:

- How much do two prompt or model variants overlap?
- Are score changes mostly due to missing coverage or true disagreements on the same filings?
- Which samples have the largest score or confidence differences?

## Environment

```bash
uv sync
```

## Basic notebook usage

```python
from pathlib import Path
from scripts.analysis import compare_results as cr

base_dir = Path("data/results/4")
datasets = [
    "4_golden_structured_input_gemi_batch",
    "4_extraction_v1_gemini",
]

result_files = cr.build_result_files_from_names(datasets, base_dir=base_dir)
artifacts = cr.prepare_comparison(result_files)

artifacts.dataset_summary
artifacts.pairwise_overlap_summary
```

## Recommended workflow

1. Inspect `dataset_summary`
2. Inspect `pairwise_overlap_summary`
3. Select one pair of runs to compare in detail
4. Review the pairwise table and top-difference samples
5. Plot diagnostics inside a notebook

Launch a notebook session with:

```bash
uv run jupyter lab
```
