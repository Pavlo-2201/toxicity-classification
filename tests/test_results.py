import json

import numpy as np
import pandas as pd
import pytest

from toxic_core import data, results, splits


class FakePredictor:
    name = "fake"

    def __init__(self, threshold=0.5, broken=False):
        self.threshold, self.broken = threshold, broken

    def predict(self, texts):
        scores = np.array([len(t) / 10 for t in texts])
        pred = (scores >= self.threshold).astype(int)
        return scores, scores / 2, 1 - pred if self.broken else pred

    def describe(self):
        return {"artifact": {"path": "fake"}, "score": "длина/10", "probability": "длина/20",
                "protocol": {"test_used_for_selection": False}, "training_manifest_sha256": splits.MANIFEST_SHA256}


PART = pd.DataFrame({data.ID_COL: [3, 1, 2, 7], data.TEXT_COL: ["ab", "abcdefg", "abcde", "x"], data.LABEL_COL: [0, 1, 1, 0]})


def test_write_text_guarded_never_replaces_silently(tmp_path):
    path = tmp_path / "r.json"
    assert results.write_text_guarded(path, "a\n") == "записан"
    assert results.write_text_guarded(path, "a\n") == "без изменений"
    with pytest.raises(results.OutputExistsError):
        results.write_text_guarded(path, "b\n")
    assert path.read_text(encoding="utf-8") == "a\n"
    assert results.write_text_guarded(path, "b\n", overwrite=True) == "перезаписан"


def test_ensure_absent(tmp_path):
    (tmp_path / "empty").mkdir()
    results.ensure_absent([tmp_path / "empty", tmp_path / "missing.json"])
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    (tmp_path / "file.json").write_text("{}")
    with pytest.raises(results.OutputExistsError, match="file.json"):
        results.ensure_absent([tmp_path / "full", tmp_path / "file.json"])
    results.ensure_absent([tmp_path / "full", tmp_path / "file.json"], overwrite=True)


def test_evaluation_has_common_schema_and_roundtrips(tmp_path):
    report, predictions = results.evaluate(FakePredictor(), "validation", PART)
    assert list(predictions.columns) == results.PREDICTION_COLUMNS
    assert predictions["pred"].tolist() == [0, 1, 1, 0]
    assert report["format"] == results.FORMAT and report["n"] == 4 and report["threshold"] == 0.5
    assert report["data"]["split_manifest_sha256"] == splits.MANIFEST_SHA256
    assert report["metrics"]["f1_toxic"] == 1.0 and "low" in report["f1_bootstrap_ci"]

    written = results.save_evaluation(report, predictions, tmp_path, "fake")
    assert set(written.values()) == {"записан"}
    assert set(results.save_evaluation(report, predictions, tmp_path, "fake").values()) == {"без изменений"}
    loaded, loaded_pred = results.load_evaluation(tmp_path, "fake", "validation")
    assert loaded["metrics"] == report["metrics"]
    pd.testing.assert_frame_equal(loaded_pred, predictions)


def test_changed_evaluation_is_not_overwritten(tmp_path):
    report, predictions = results.evaluate(FakePredictor(), "validation", PART)
    results.save_evaluation(report, predictions, tmp_path, "fake")
    other_report, other_pred = results.evaluate(FakePredictor(threshold=0.3), "validation", PART)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(results.OutputExistsError):
        results.save_evaluation(other_report, other_pred, tmp_path, "fake")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before  # ни один из двух файлов не тронут


def test_decisions_must_match_threshold():
    with pytest.raises(RuntimeError):
        results.evaluate(FakePredictor(broken=True), "validation", PART)


def test_tampered_predictions_are_detected(tmp_path):
    report, predictions = results.evaluate(FakePredictor(), "validation", PART)
    results.save_evaluation(report, predictions, tmp_path, "fake")
    path = tmp_path / "fake_validation_predictions.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",1\n", ",0\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        results.load_evaluation(tmp_path, "fake", "validation")


def test_compare_requires_same_texts_and_data():
    a = results.evaluate(FakePredictor(), "test", PART)
    b = results.evaluate(FakePredictor(threshold=0.15), "test", PART.iloc[::-1])  # порядок строк не важен
    report = results.compare(a, b, ("a", "b"))
    assert report["n"] == 4 and report["mcnemar"]["only_first_correct"] == 1
    assert report["f1_toxic_difference"]["point"] == round(b[0]["metrics"]["f1_toxic"] - 1.0, 4)

    with pytest.raises(ValueError, match="source_row"):
        results.compare(a, results.evaluate(FakePredictor(), "test", PART.head(3)), ("a", "c"))
    other_data = ({**a[0], "data": {**a[0]["data"], "split_manifest_sha256": "0" * 64}}, a[1])
    with pytest.raises(ValueError, match="разные данные"):
        results.compare(a, other_data, ("a", "d"))
    with pytest.raises(ValueError, match="разные разбиения"):
        results.compare(a, results.evaluate(FakePredictor(), "validation", PART), ("a", "v"))


def test_report_is_json_serializable():
    report, _ = results.evaluate(FakePredictor(), "test", PART, limit=4)
    assert json.loads(results.json_text(report))["limit"] == 4
