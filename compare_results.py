from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path(r"D:\AQUMON\data\results\4")


@dataclass(frozen=True)
class ComparisonArtifacts:
    result_tables: dict[str, pd.DataFrame]
    dataset_summary: pd.DataFrame
    pairwise_overlap_summary: pd.DataFrame
    comparison_tables: dict[tuple[str, str], pd.DataFrame]


def confidence_to_float(value) -> float:
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


def read_results_table(path: str | Path) -> pd.DataFrame:
    source = resolve_results_source(path)

    if source.suffix == ".parquet":
        try:
            df = pd.read_parquet(source)
        except Exception as exc:
            fallback_jsonl = source.with_suffix(".jsonl")
            if not fallback_jsonl.exists():
                raise RuntimeError(
                    f"Failed to read parquet file {source} and no jsonl fallback was found."
                ) from exc
            source = fallback_jsonl
            records = []
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            df = pd.DataFrame(records)
    elif source.suffix == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported source type: {source}")

    required_columns = {"id", "sentiment_score", "confidence"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns {missing_columns} in {source}")

    standardized = df.copy()
    standardized["id"] = standardized["id"].astype(str)
    standardized["sentiment_score"] = pd.to_numeric(
        standardized["sentiment_score"], errors="coerce"
    )
    standardized["confidence_pct"] = standardized["confidence"].map(confidence_to_float)
    if "summary" not in standardized.columns:
        standardized["summary"] = ""
    standardized["source_path"] = str(source)
    return standardized


def load_all_results(result_files: Mapping[str, str | Path]) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in result_files.items():
        loaded[name] = read_results_table(path)
    return loaded


def build_dataset_summary(result_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in result_tables.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "unique_ids": df["id"].nunique(),
                "score_non_null": df["sentiment_score"].notna().sum(),
                "confidence_non_null": df["confidence_pct"].notna().sum(),
                "score_mean": df["sentiment_score"].mean(),
                "confidence_mean": df["confidence_pct"].mean(),
                "source_path": df["source_path"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def build_pairwise_overlap_summary(
    result_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    id_sets = {name: set(df["id"]) for name, df in result_tables.items()}

    for left_name, right_name in combinations(result_tables.keys(), 2):
        left_ids = id_sets[left_name]
        right_ids = id_sets[right_name]
        overlap = left_ids & right_ids
        union = left_ids | right_ids
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "left_rows": len(left_ids),
                "right_rows": len(right_ids),
                "intersection_rows": len(overlap),
                "union_rows": len(union),
                "jaccard": len(overlap) / len(union) if union else np.nan,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["intersection_rows", "left", "right"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def normalize_pairs(
    dataset_names: Sequence[str],
    pairs_to_compare: Optional[Sequence[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    dataset_name_set = set(dataset_names)

    if pairs_to_compare is None:
        return list(combinations(dataset_names, 2))

    normalized: list[tuple[str, str]] = []
    for left_name, right_name in pairs_to_compare:
        if left_name not in dataset_name_set:
            raise KeyError(f"Unknown dataset name in pair: {left_name}")
        if right_name not in dataset_name_set:
            raise KeyError(f"Unknown dataset name in pair: {right_name}")
        normalized.append((left_name, right_name))
    return normalized


def compare_pair(
    result_tables: Mapping[str, pd.DataFrame],
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    left = result_tables[left_name][
        ["id", "sentiment_score", "confidence", "confidence_pct", "summary"]
    ].copy()
    right = result_tables[right_name][
        ["id", "sentiment_score", "confidence", "confidence_pct", "summary"]
    ].copy()

    merged = left.merge(
        right,
        on="id",
        how="inner",
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )

    merged["sentiment_score_diff"] = (
        merged[f"sentiment_score_{right_name}"] - merged[f"sentiment_score_{left_name}"]
    )
    merged["sentiment_score_abs_diff"] = merged["sentiment_score_diff"].abs()
    merged["confidence_pct_diff"] = (
        merged[f"confidence_pct_{right_name}"] - merged[f"confidence_pct_{left_name}"]
    )
    merged["confidence_pct_abs_diff"] = merged["confidence_pct_diff"].abs()
    return (
        merged.sort_values(
            ["sentiment_score_abs_diff", "confidence_pct_abs_diff"], ascending=False
        )
        .reset_index(drop=True)
    )


def build_comparison_tables(
    result_tables: Mapping[str, pd.DataFrame],
    pairs_to_compare: Optional[Sequence[tuple[str, str]]] = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    pairs = normalize_pairs(list(result_tables.keys()), pairs_to_compare)
    return {pair: compare_pair(result_tables, *pair) for pair in pairs}


def build_pair_summary_stats(
    pair_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    intersection_rows = len(pair_df)
    sentiment_corr = (
        pair_df[
            [f"sentiment_score_{left_name}", f"sentiment_score_{right_name}"]
        ].corr().iloc[0, 1]
        if intersection_rows >= 2
        else np.nan
    )
    confidence_corr = (
        pair_df[[f"confidence_pct_{left_name}", f"confidence_pct_{right_name}"]]
        .corr()
        .iloc[0, 1]
        if intersection_rows >= 2
        else np.nan
    )

    return pd.DataFrame(
        [
            {"metric": "intersection_rows", "value": intersection_rows},
            {
                "metric": f"mean_sentiment_score_{left_name}",
                "value": pair_df[f"sentiment_score_{left_name}"].mean(),
            },
            {
                "metric": f"mean_sentiment_score_{right_name}",
                "value": pair_df[f"sentiment_score_{right_name}"].mean(),
            },
            {
                "metric": "mean_sentiment_score_abs_diff",
                "value": pair_df["sentiment_score_abs_diff"].mean(),
            },
            {
                "metric": f"mean_confidence_pct_{left_name}",
                "value": pair_df[f"confidence_pct_{left_name}"].mean(),
            },
            {
                "metric": f"mean_confidence_pct_{right_name}",
                "value": pair_df[f"confidence_pct_{right_name}"].mean(),
            },
            {
                "metric": "mean_confidence_pct_abs_diff",
                "value": pair_df["confidence_pct_abs_diff"].mean(),
            },
            {"metric": "sentiment_score_corr", "value": sentiment_corr},
            {"metric": "confidence_pct_corr", "value": confidence_corr},
        ]
    )


def get_top_differences(
    pair_df: pd.DataFrame,
    left_name: str,
    right_name: str,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    top_score_diffs = pair_df.nlargest(top_n, "sentiment_score_abs_diff")[
        [
            "id",
            f"sentiment_score_{left_name}",
            f"sentiment_score_{right_name}",
            "sentiment_score_diff",
            f"confidence_pct_{left_name}",
            f"confidence_pct_{right_name}",
            f"summary_{left_name}",
            f"summary_{right_name}",
        ]
    ]

    top_confidence_diffs = pair_df.nlargest(top_n, "confidence_pct_abs_diff")[
        [
            "id",
            f"confidence_pct_{left_name}",
            f"confidence_pct_{right_name}",
            "confidence_pct_diff",
            f"sentiment_score_{left_name}",
            f"sentiment_score_{right_name}",
            f"summary_{left_name}",
            f"summary_{right_name}",
        ]
    ]

    return {
        "top_score_diffs": top_score_diffs,
        "top_confidence_diffs": top_confidence_diffs,
    }


def plot_pair_diagnostics(
    pair_df: pd.DataFrame,
    left_name: str,
    right_name: str,
    *,
    show: bool = True,
) -> dict[str, Optional[plt.Figure]]:
    if pair_df.empty:
        return {"main": None, "heatmap": None}

    fig_main, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].scatter(
        pair_df[f"sentiment_score_{left_name}"],
        pair_df[f"sentiment_score_{right_name}"],
        alpha=0.5,
        s=18,
        edgecolor="none",
    )
    score_values = pd.concat(
        [
            pair_df[f"sentiment_score_{left_name}"],
            pair_df[f"sentiment_score_{right_name}"],
        ]
    ).dropna()
    if not score_values.empty:
        score_min = score_values.min()
        score_max = score_values.max()
        axes[0, 0].plot(
            [score_min, score_max],
            [score_min, score_max],
            linestyle="--",
            color="black",
            linewidth=1,
        )
    axes[0, 0].set_title("Sentiment score scatter")
    axes[0, 0].set_xlabel(left_name)
    axes[0, 0].set_ylabel(right_name)

    axes[0, 1].hist(
        pair_df["sentiment_score_diff"].dropna(),
        bins=min(30, max(1, len(pair_df))),
        color="#4C78A8",
        alpha=0.85,
    )
    axes[0, 1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[0, 1].set_title("Sentiment score difference distribution")
    axes[0, 1].set_xlabel(f"{right_name} - {left_name}")

    axes[1, 0].scatter(
        pair_df[f"confidence_pct_{left_name}"],
        pair_df[f"confidence_pct_{right_name}"],
        alpha=0.5,
        s=18,
        edgecolor="none",
        color="#F58518",
    )
    conf_values = pd.concat(
        [
            pair_df[f"confidence_pct_{left_name}"],
            pair_df[f"confidence_pct_{right_name}"],
        ]
    ).dropna()
    if not conf_values.empty:
        conf_min = conf_values.min()
        conf_max = conf_values.max()
        axes[1, 0].plot(
            [conf_min, conf_max],
            [conf_min, conf_max],
            linestyle="--",
            color="black",
            linewidth=1,
        )
    axes[1, 0].set_title("Confidence scatter")
    axes[1, 0].set_xlabel(left_name)
    axes[1, 0].set_ylabel(right_name)

    axes[1, 1].hist(
        pair_df["confidence_pct_diff"].dropna(),
        bins=min(30, max(1, len(pair_df))),
        color="#E45756",
        alpha=0.85,
    )
    axes[1, 1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1, 1].set_title("Confidence difference distribution")
    axes[1, 1].set_xlabel(f"{right_name} - {left_name}")

    plt.tight_layout()
    if show:
        plt.show()

    fig_heatmap: Optional[plt.Figure] = None
    if len(pair_df) >= 2:
        corr_matrix = pair_df[
            [
                f"sentiment_score_{left_name}",
                f"sentiment_score_{right_name}",
                f"confidence_pct_{left_name}",
                f"confidence_pct_{right_name}",
            ]
        ].corr()

        fig_heatmap, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.index)

        for row_idx in range(corr_matrix.shape[0]):
            for col_idx in range(corr_matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{corr_matrix.iloc[row_idx, col_idx]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_title(f"Correlation heatmap on intersection: {left_name} vs {right_name}")
        fig_heatmap.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if show:
            plt.show()

    return {"main": fig_main, "heatmap": fig_heatmap}


def prepare_comparison(
    result_files: Mapping[str, str | Path],
    pairs_to_compare: Optional[Sequence[tuple[str, str]]] = None,
) -> ComparisonArtifacts:
    result_tables = load_all_results(result_files)
    dataset_summary = build_dataset_summary(result_tables)
    pairwise_overlap_summary = build_pairwise_overlap_summary(result_tables)
    comparison_tables = build_comparison_tables(result_tables, pairs_to_compare)
    return ComparisonArtifacts(
        result_tables=result_tables,
        dataset_summary=dataset_summary,
        pairwise_overlap_summary=pairwise_overlap_summary,
        comparison_tables=comparison_tables,
    )


def build_result_files_from_names(
    dataset_names: Sequence[str],
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> dict[str, Path]:
    return {name: base_dir / name for name in dataset_names}
