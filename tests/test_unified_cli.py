"""Сквозная проверка общего интерфейса `toxic` на классической модели (без PyTorch).

Артефакт — маленькая модель на игрушечных текстах, сохранённая так же, как это делает
classical/train.py; оценка идёт по настоящим разбиениям проекта.
"""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toxic_core import data, results, splits

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
needs_data = pytest.mark.skipif(not data.DATASET_FILE.exists(), reason="данные не подготовлены: toxic prepare")

TOXIC = ["ты тупой идиот", "заткнись, дебил!!!", "иди нафиг, урод", "какой же ты идиот"]
CLEAN = ["спасибо за статью", "хороший день сегодня", "интересная мысль, согласен", "поддерживаю автора"]
CONFIG = {"name": "toy", "model": "linearsvc", "model_params": {"C": 0.3, "class_weight": "balanced"},
          "char": {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 1}}


def toxic(*args, check=True):
    out = subprocess.run([sys.executable, "-m", "toxic_core", *map(str, args)], capture_output=True, text=True,
                         encoding="utf-8", env=ENV, cwd=ROOT)
    if check and out.returncode:
        raise AssertionError(f"toxic {' '.join(map(str, args))}: {out.returncode}\n{out.stdout}\n{out.stderr}")
    return out


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    from toxic_clf.calibration import PlattCalibrator
    from toxic_clf.final_model import build_final_model, save_model, toxicity_scores

    texts, labels = (TOXIC + CLEAN) * 3, [1, 1, 1, 1, 0, 0, 0, 0] * 3
    model = build_final_model(CONFIG, threshold=0.05).fit(texts, labels)
    calibrator = PlattCalibrator.fit(toxicity_scores(model, texts), labels)
    path = tmp_path_factory.mktemp("artifact") / "toy.joblib"
    save_model(model, {"config": CONFIG, "normalization": "baseline", "threshold": 0.05,
                       "threshold_selection": {"used": 0.05, "rule": "игрушечный артефакт для теста"},
                       "calibration": calibrator.to_dict(), "manifest_sha256": splits.MANIFEST_SHA256}, path)
    return path


def test_classical_path_does_not_import_torch():
    code = ("import sys, toxic_core.cli, toxic_core.results, toxic_clf.adapter, toxic_clf.final_model; "
            "assert 'torch' not in sys.modules, 'torch импортирован'")
    subprocess.run([sys.executable, "-c", code], check=True, env=ENV)


def test_help_lists_commands():
    out = toxic("--help").stdout
    for command in ("prepare", "train", "predict", "evaluate", "compare", "check"):
        assert command in out


def test_predict_matches_loaded_model(artifact, tmp_path):
    from toxic_clf.adapter import load

    texts = ["ну ты и идиот", "отличная статья, спасибо 😀"]
    out = toxic("predict", "classical", "--artifact", artifact, *[a for t in texts for a in ("--text", t)])
    df = pd.read_csv(io.StringIO(out.stdout), keep_default_na=False)
    assert list(df.columns) == ["comment", "score", "probability", "toxic"] and df["comment"].tolist() == texts
    scores, probs, pred = load(artifact).predict(texts)
    np.testing.assert_allclose(df["score"], scores.round(4), atol=1e-9)
    np.testing.assert_allclose(df["probability"], probs.round(4), atol=1e-9)
    assert df["toxic"].tolist() == pred.tolist()

    source, output = tmp_path / "in.txt", tmp_path / "out.csv"
    source.write_text("﻿" + "\n".join(texts) + "\n\n", encoding="utf-8")
    toxic("predict", "classical", "--artifact", artifact, "--input", source, "--output", output)
    pd.testing.assert_frame_equal(pd.read_csv(output, keep_default_na=False), df)
    again = toxic("predict", "classical", "--artifact", artifact, "--input", source, "--output", output, check=False)
    assert again.returncode != 0 and "--overwrite" in again.stderr


@needs_data
def test_evaluate_compare_and_no_silent_overwrite(artifact, tmp_path):
    out_dir = tmp_path / "eval"
    toxic("evaluate", "classical", "--artifact", artifact, "--split", "validation", "--limit", 300, "--out-dir", out_dir)
    report, pred = results.load_evaluation(out_dir, "classical-toy-first300", "validation")
    validation = splits.load_splits()["validation"].head(300)
    assert pred[data.ID_COL].tolist() == validation[data.ID_COL].tolist()
    assert pred[data.LABEL_COL].tolist() == validation[data.LABEL_COL].tolist()
    assert report["threshold"] == 0.05 and report["n"] == 300 and report["limit"] == 300

    # повторная оценка того же артефакта — те же байты; отличающийся результат не заменяет готовый
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir.iterdir()}
    assert "без изменений" in toxic("evaluate", "classical", "--artifact", artifact, "--split", "validation",
                                    "--limit", 300, "--out-dir", out_dir).stdout
    changed = toxic("evaluate", "classical", "--artifact", artifact, "--split", "validation", "--limit", 200,
                    "--name", "classical-toy-first300", "--out-dir", out_dir, check=False)
    assert changed.returncode != 0 and "--overwrite" in changed.stderr
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir.iterdir()} == before

    toxic("evaluate", "classical", "--artifact", artifact, "--split", "validation", "--limit", 300,
          "--name", "copy", "--out-dir", out_dir)
    toxic("compare", "classical-toy-first300", "copy", "--split", "validation", "--out-dir", out_dir)
    compared = json.loads((out_dir / "compare-classical-toy-first300-vs-copy_validation.json").read_text(encoding="utf-8"))
    assert compared["n"] == 300 and compared["decisions"]["agree"] == 300 and compared["mcnemar"]["p_value"] == 1.0


@needs_data
def test_evaluate_refuses_artifact_from_other_manifest(artifact, tmp_path):
    other = tmp_path / "other.joblib"
    other.write_bytes(artifact.read_bytes())
    meta = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    other.with_suffix(".json").write_text(json.dumps({**meta, "manifest_sha256": "0" * 64}), encoding="utf-8")
    out = toxic("evaluate", "classical", "--artifact", other, "--out-dir", tmp_path / "eval", check=False)
    assert out.returncode != 0 and "манифест" in out.stderr and not (tmp_path / "eval").exists()


def test_train_refuses_to_overwrite_published_reports():
    """В клоне отчёты classical закоммичены: обучение без --overwrite останавливается до начала работы."""
    published = [data.REPO_ROOT / "classical" / "reports" / name for name in ("final_metrics.json", "test_predictions.csv")]
    before = [p.read_bytes() for p in published]
    out = toxic("train", "classical", check=False)
    assert out.returncode != 0 and "--overwrite" in out.stderr
    assert [p.read_bytes() for p in published] == before


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="проверка установки без PyTorch")
def test_rubert_without_torch_gives_install_hint():
    out = toxic("predict", "rubert", "--text", "привет", check=False)
    assert out.returncode != 0 and "requirements/rubert.txt" in out.stderr


@needs_data
def test_check_reports_consistent_data_and_artifacts():
    out = toxic("check")
    assert '"passed": true' in out.stdout
