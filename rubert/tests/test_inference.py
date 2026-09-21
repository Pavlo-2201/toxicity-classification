import json
import shutil
import subprocess
import sys

import numpy as np
import torch

from rubert_clf import data, model_source
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.finetune import FinetuneConfig, build_model, fit
from rubert_clf.inference import ToxicityClassifier
from rubert_clf.runtime import set_seed

TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья"]
LABELS = [1, 0, 1, 0, 1, 0]


def _trained_dir(tokenizer, tmp_path):
    set_seed(0)
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=16)
    history = fit(build_model(model_source.MODEL_DIR), ds, ds, tokenizer,
                  FinetuneConfig(epochs=2, batch_size=3, lr=1e-4, max_length=16), torch.device("cpu"),
                  checkpoint_dir=tmp_path, log=lambda *_: None)
    # last.pt (веса + оптимизатор, ~350 МБ) нужен только для продолжения обучения; каталоги тестов pytest
    # удаляет лишь в конце сессии, и без этого файлы копятся до сотен мегабайт на тест
    (tmp_path / "last.pt").unlink()
    return tmp_path / "best", history


def test_saved_model_reproduces_validation_probabilities(tokenizer, tmp_path):
    model_dir, history = _trained_dir(tokenizer, tmp_path)
    clf = ToxicityClassifier.load(model_dir, "cpu")
    best = history[clf.selection["epoch"] - 1]
    probs, expected = clf.predict_proba(TEXTS), np.asarray(best["val_probs"])
    np.testing.assert_allclose(probs, expected, atol=1e-5)
    assert clf.max_length == 16 and clf.threshold == best["val_threshold"]
    np.testing.assert_array_equal(clf.predict(TEXTS), (probs >= clf.threshold).astype(int))
    # порог — одна из вероятностей validation (precision_recall_curve): у этого текста решение зависит
    # от последних битов пересчёта, поэтому с обучением сверяются только тексты не на самом пороге
    away = np.abs(expected - clf.threshold) > 1e-5
    np.testing.assert_array_equal(clf.predict(TEXTS)[away], (expected[away] >= clf.threshold).astype(int))
    assert clf.predict_proba([]).shape == (0,)


def test_predict_cli(tokenizer, tmp_path):
    model_dir, _ = _trained_dir(tokenizer, tmp_path)
    out = subprocess.run([sys.executable, str(data.PROJECT_ROOT / "predict.py"), "--model-dir", str(model_dir),
                          "--device", "cpu", *TEXTS[:2], "Спасибо 😀"], capture_output=True, text=True, encoding="utf-8",
                         check=True)
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    assert [r["text"] for r in rows] == [*TEXTS[:2], "Спасибо 😀"]
    assert all(0 <= r["toxic_probability"] <= 1 and r["toxic"] in (0, 1) for r in rows)


def test_train_cli_smoke(dataset, tokenizer, tmp_path):
    subprocess.run([sys.executable, str(data.PROJECT_ROOT / "train.py"), "--smoke", "--device", "cpu",
                    "--output-dir", str(tmp_path / "model"), "--checkpoint-dir", str(tmp_path / "ckpt")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    shutil.rmtree(tmp_path / "ckpt")  # чекпойнты продолжения обучения тесту не нужны
    meta = json.loads((tmp_path / "model" / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "smoke" and meta["n_train"] == 256
    assert ToxicityClassifier.load(tmp_path / "model", "cpu").predict(["Ты дурак!"]).shape == (1,)
