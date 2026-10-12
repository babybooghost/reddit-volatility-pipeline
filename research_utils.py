"""Validated inputs and evaluation for volatility research."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

FloatArray = NDArray[np.float64]
FrameSource = pd.DataFrame | str | Path


def read_frame(source: FrameSource) -> pd.DataFrame:
    """Read a local CSV, JSONL, or parquet table without mutating input frames."""
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(
            path,
            dtype={"firm_id": "string", "cik": "string", "gvkey": "string", "post_id": "string"},
        )
    raise ValueError(f"Unsupported table format: {path.suffix}")


def require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    """Reject empty tables and incomplete schemas before downstream work."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.empty:
        raise ValueError("Input table is empty")


def utc(values: pd.Series) -> pd.Series:
    """Parse explicit UTC timestamps. Reject missing values."""
    result = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
    if result.isna().any():
        raise ValueError("Timestamps cannot be missing")
    return result


def cutoff_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize a nonmissing temporal boundary to UTC."""
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Cutoff cannot be missing")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def matrix(values: Any, *, width: int | None = None) -> FloatArray:
    """Validate a finite dense feature matrix with an optional fixed width."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite two-dimensional matrix")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"Expected {width} columns, got {result.shape[1]}")
    return result


def binary_labels(values: Any, count: int, *, both_classes: bool = True) -> NDArray[np.int64]:
    """Validate binary outcomes and their alignment with feature rows."""
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != count or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be aligned binary values without missing entries")
    if both_classes and len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required for training and evaluation")
    return labels.astype(np.int64)


def classification_report(y: Any, scores: Any, *, threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate continuous scores and fixed-threshold binary predictions."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    labels = binary_labels(y, len(scores))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "pr_auc_definition": "average_precision",
        "prevalence": float(labels.mean()),
        "confusion_matrix": confusion_matrix(labels, scores >= threshold, labels=[0, 1]).tolist(),
        "confusion_matrix_order": [["TN", "FP"], ["FN", "TP"]],
        "decision_threshold": threshold,
        "samples": len(labels),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a standards-compliant JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
