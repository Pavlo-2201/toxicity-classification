"""Протокол train.py: порог и калибровка выбираются по OOF train и validation, test на них не влияет.

train.py запускается целиком (конфигурация из reports/best_model.json, вариант нормализации all, чтобы
не требовать совпадения порога с отчётом baseline) на игрушечных разбиениях: два прогона отличаются
только метками test.
"""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from toxic_clf import data
from toxic_clf.final_model import load_calibrator, load_model, toxicity_scores
from toxic_core import splits

spec = importlib.util.spec_from_file_location("classical_train", data.PROJECT_ROOT / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)

INSULTS = ["идиот", "дебил", "урод", "тупица", "придурок", "кретин"]
NEUTRAL = ["статья", "мысль", "идея", "история", "новость", "работа"]


def toy_parts() -> dict:
    rows = [(f"ты {w} {i}", 1) for i in range(4) for w in INSULTS] + [(f"хорошая {w} {i}", 0) for i in range(6) for w in NEUTRAL]
    df = pd.DataFrame(rows, columns=[data.TEXT_COL, data.LABEL_COL]).sample(frac=1, random_state=0).reset_index(drop=True)
    df.insert(0, data.ID_COL, range(len(df)))
    return {"train": df.iloc[:40].reset_index(drop=True), "validation": df.iloc[40:50].reset_index(drop=True),
            "test": df.iloc[50:].reset_index(drop=True)}


def train(tmp_path, monkeypatch, parts, name: str):
    monkeypatch.setattr(splits, "load_splits", lambda *a, **k: parts)
    output, reports = tmp_path / name / "model.joblib", tmp_path / name / "reports"
    monkeypatch.setattr(sys, "argv", ["train.py", "--normalization", "all", "--output", str(output),
                                      "--reports-dir", str(reports)])
    train_script.main()
    return output, json.loads((reports / "final_metrics.json").read_text(encoding="utf-8"))


def test_test_labels_do_not_affect_threshold_or_model(tmp_path, monkeypatch):
    parts = toy_parts()
    flipped = {**parts, "test": parts["test"].assign(**{data.LABEL_COL: 1 - parts["test"][data.LABEL_COL]})}
    path_a, metrics_a = train(tmp_path, monkeypatch, parts, "a")
    path_b, metrics_b = train(tmp_path, monkeypatch, flipped, "b")

    assert metrics_a["threshold_selection"] == metrics_b["threshold_selection"]
    assert metrics_a["validation"] == metrics_b["validation"]
    assert metrics_a["test"]["f1_toxic"] != metrics_b["test"]["f1_toxic"]  # test действительно другой
    texts = parts["test"][data.TEXT_COL].tolist() + ["ну ты и идиот", "интересная новость"]
    model_a, model_b = load_model(path_a), load_model(path_b)
    assert model_a.threshold == model_b.threshold
    np.testing.assert_array_equal(toxicity_scores(model_a, texts), toxicity_scores(model_b, texts))
    assert load_calibrator(path_a) == load_calibrator(path_b)


def test_existing_outputs_are_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "model.joblib"
    output.write_bytes(b"previous")
    monkeypatch.setattr(splits, "load_splits", lambda *a, **k: pytest.fail("обучение не должно начаться"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--normalization", "all", "--output", str(output),
                                      "--reports-dir", str(tmp_path / "r")])
    with pytest.raises(SystemExit, match="--overwrite"):
        train_script.main()
    assert output.read_bytes() == b"previous"
