"""Input serialization and metric contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from research_utils import binary_labels, classification_report, read_frame, write_json


@pytest.mark.parametrize("extension", ["csv", "parquet", "jsonl"])
def test_read_frame_roundtrip(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"data.{extension}"
    frame = pd.DataFrame({"firm_id": ["firm_001", "firm_002"], "x": [1.5, 2.5]})
    if extension == "csv":
        frame.to_csv(path, index=False)
    elif extension == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_json(path, orient="records", lines=True)
    result = read_frame(path)
    assert result.firm_id.tolist() == frame.firm_id.tolist()
    np.testing.assert_array_equal(result.x, frame.x)


def test_csv_preserves_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("firm_id,x\n001,2\n", encoding="utf-8")
    assert read_frame(path).firm_id.iloc[0] == "001"


def test_atomic_json_retains_previous_report_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"ok": True})
    with pytest.raises(ValueError):
        write_json(path, {"bad": np.nan})
    assert json.loads(path.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_metrics_use_scores_and_fixed_confusion_order() -> None:
    y, scores = [0, 1, 0, 1], [0.1, 0.9, 0.8, 0.3]
    report = classification_report(y, scores)
    assert report["roc_auc"] == roc_auc_score(y, scores)
    assert report["pr_auc"] == average_precision_score(y, scores)
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]


@pytest.mark.parametrize("values", [[0, 2], [0, np.nan], [[0, 1]], [0]])
def test_invalid_binary_labels(values: list) -> None:
    with pytest.raises(ValueError):
        binary_labels(values, 2)
