import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from toxic_clf.calibration import PlattCalibrator
from toxic_clf.final_model import (build_final_model, load_calibrator, load_metadata, load_model, save_model,
                                   toxicity_scores)
from toxic_clf.selection import choose_threshold

ROOT = Path(__file__).resolve().parents[1]

TOXIC = ["ты тупой идиот", "заткнись, дебил!!!", "иди нафиг, урод", "какой же ты идиот"]
CLEAN = ["спасибо за статью", "хороший день сегодня", "интересная мысль, согласен", "поддерживаю автора"]
TEXTS = TOXIC * 3 + CLEAN * 3
LABELS = [1] * 12 + [0] * 12
CONFIG = {
    "name": "t",
    "model": "linearsvc",
    "model_params": {"C": 0.3, "class_weight": "balanced"},
    "char": {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 1},
}
NEW = ["ну ты и идиот", "отличная статья, спасибо", "  Заткнись!!!\n"]


def test_threshold_is_applied_to_decision_scores():
    model = build_final_model(CONFIG, threshold=0.25).fit(TEXTS, LABELS)
    scores = toxicity_scores(model, TEXTS + NEW)
    assert not hasattr(model, "predict_proba")
    assert (model.predict(TEXTS + NEW) == (scores >= 0.25).astype(int)).all()


def test_save_load_roundtrip_gives_identical_predictions(tmp_path):
    model = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    path = tmp_path / "model.joblib"
    save_model(model, {"config": CONFIG}, path)
    loaded = load_model(path)
    assert (path.with_suffix(".json")).exists()
    np.testing.assert_array_equal(loaded.predict(NEW), model.predict(NEW))
    np.testing.assert_allclose(toxicity_scores(loaded, NEW), toxicity_scores(model, NEW))


def test_training_is_reproducible():
    a = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    b = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    np.testing.assert_allclose(toxicity_scores(a, NEW), toxicity_scores(b, NEW))


ALL_CONFIG = {**CONFIG, "name": "t|all", "normalization": "all"}
# латиница-двойник, растяжка и маска: нормализация "all" должна сохраниться в артефакте
DISGUISED = ["ты тупoй идиoт", "заткниииись, д*бил", "спасибо за стaтью"]


def saved_bundle(tmp_path):
    model = build_final_model(ALL_CONFIG, threshold=0.1).fit(TEXTS, LABELS)
    calibrator = PlattCalibrator.fit(toxicity_scores(model, TEXTS), LABELS)
    path = tmp_path / "bundle.joblib"
    meta = {"config": ALL_CONFIG, "normalization": "all", "threshold": 0.1, "calibration": calibrator.to_dict()}
    save_model(model, meta, path)
    return model, calibrator, path


def test_saved_bundle_keeps_normalization_threshold_and_calibrator(tmp_path):
    model, calibrator, path = saved_bundle(tmp_path)
    loaded, loaded_calibrator = load_model(str(path)), load_calibrator(str(path))  # строка — как из argparse
    texts = NEW + DISGUISED
    scores = toxicity_scores(loaded, texts)
    np.testing.assert_array_equal(scores, toxicity_scores(model, texts))
    assert loaded.threshold == 0.1 and load_metadata(path)["normalization"] == "all"
    np.testing.assert_array_equal(loaded.predict(texts), (scores >= 0.1).astype(int))
    assert loaded_calibrator == calibrator
    np.testing.assert_array_equal(loaded_calibrator.predict_proba(scores), calibrator.predict_proba(scores))
    np.testing.assert_allclose(toxicity_scores(loaded, ["ты тупoй идиoт"]), toxicity_scores(loaded, ["ты тупой идиот"]))


def test_predict_cli_matches_loaded_bundle(tmp_path):
    _, _, path = saved_bundle(tmp_path)
    texts = NEW[:2] + DISGUISED
    source, output = tmp_path / "in.csv", tmp_path / "out.csv"
    pd.DataFrame({"comment": texts}).to_csv(source, index=False)
    subprocess.run([sys.executable, str(ROOT / "predict.py"), "--model", str(path), "--input", str(source),
                    "--output", str(output)], check=True, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    pred = pd.read_csv(output, keep_default_na=False)
    loaded, calibrator = load_model(path), load_calibrator(path)
    scores = toxicity_scores(loaded, texts)
    assert pred["comment"].tolist() == texts
    np.testing.assert_allclose(pred["score"], scores.round(4), atol=1e-9)
    np.testing.assert_allclose(pred["probability"], calibrator.predict_proba(scores).round(4), atol=1e-9)
    np.testing.assert_array_equal(pred["toxic"], (scores >= loaded.threshold).astype(int))


def test_choose_threshold_keeps_default_when_validation_is_worse():
    y_train, oof = np.array([0, 0, 1, 1]), np.array([-1.0, -0.2, -0.1, 1.0])  # OOF-оптимум: -0.1
    y_val = np.array([0, 1, 0, 1])
    worse = choose_threshold(CONFIG, oof, y_train, np.array([-0.05, 0.5, -2.0, 0.3]), y_val)
    assert worse["oof_tuned"] == -0.1 and worse["used"] == 0.0
    same = choose_threshold(CONFIG, oof, y_train, np.array([-0.5, 0.5, -2.0, 0.3]), y_val)
    assert same["used"] == -0.1
