"""Общий интерфейс `toxic` для RuBERT: лёгкий smoke (rubert-tiny2, 256 текстов, 1 эпоха на CPU).

Полное обучение по протоколу в тестах слишком дорого; здесь проверяется путь целиком:
train → predict → evaluate в общем формате, и что готовый артефакт не перезаписывается.
"""

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toxic_core import results, splits

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def toxic(*args, check=True):
    out = subprocess.run([sys.executable, "-m", "toxic_core", *map(str, args)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=ENV, cwd=ROOT)
    if check and out.returncode:
        raise AssertionError(f"toxic {' '.join(map(str, args))}: {out.returncode}\n{out.stdout}\n{out.stderr}")
    return out


@pytest.fixture(scope="module")
def smoke_model(tmp_path_factory, dataset, tokenizer):
    tmp = tmp_path_factory.mktemp("smoke")
    toxic("train", "rubert", "--smoke", "--device", "cpu", "--output-dir", tmp / "model", "--checkpoint-dir", tmp / "ckpt")
    shutil.rmtree(tmp / "ckpt")  # чекпойнты продолжения обучения (~0.5 ГБ) тестам не нужны
    return tmp / "model"


def test_predict_returns_common_columns(smoke_model):
    from rubert_clf.inference import ToxicityClassifier

    texts = ["Ты дурак!", "Спасибо 😀"]
    out = toxic("predict", "rubert", "--artifact", smoke_model, "--device", "cpu", "--text", texts[0], "--text", texts[1])
    df = pd.read_csv(io.StringIO(out.stdout), keep_default_na=False)
    assert list(df.columns) == ["comment", "score", "probability", "toxic"] and df["comment"].tolist() == texts
    clf = ToxicityClassifier.load(smoke_model, "cpu")
    probs = clf.predict_proba(texts)
    np.testing.assert_allclose(df["probability"], probs.round(4), atol=1e-9)
    assert (df["score"] == df["probability"]).all() and df["toxic"].tolist() == (probs >= clf.threshold).astype(int).tolist()


def test_evaluate_in_common_format(smoke_model, tmp_path):
    toxic("evaluate", "rubert", "--artifact", smoke_model, "--split", "validation", "--limit", 64, "--device", "cpu",
          "--out-dir", tmp_path)
    report, pred = results.load_evaluation(tmp_path, "rubert-model-first64", "validation")
    assert report["model"] == "rubert" and report["n"] == 64 and list(pred.columns) == results.PREDICTION_COLUMNS
    assert report["data"]["split_manifest_sha256"] == splits.MANIFEST_SHA256
    assert report["protocol"]["test_used_for_selection"] is False
    selection = json.loads((smoke_model / "selection.json").read_text(encoding="utf-8"))
    assert report["threshold"] == selection["threshold"] and report["artifact"]["epoch"] == selection["epoch"]
    assert (pred["pred"] == (pred["score"] >= selection["threshold"]).astype(int)).all()


def test_train_does_not_overwrite_existing_artifact(smoke_model):
    before = {p.name: p.read_bytes() for p in smoke_model.iterdir() if p.is_file()}
    out = toxic("train", "rubert", "--smoke", "--device", "cpu", "--output-dir", smoke_model, check=False)
    assert out.returncode != 0 and "--overwrite" in out.stderr
    assert {p.name: p.read_bytes() for p in smoke_model.iterdir() if p.is_file()} == before
