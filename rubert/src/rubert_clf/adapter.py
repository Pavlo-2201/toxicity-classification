"""RuBERT в общем интерфейсе (toxic_core.registry).

score и probability — P(toxic) (softmax), порог артефакта применяется к ней. Протокол: эпоха —
по AP на validation, порог — F1-оптимум на калибровочной части train (15 % train, модель на ней не
обучается). Test в выборе не участвует. Артефакт при загрузке сверяется с отчётом своего прогона
(rubert_clf.inference.verify_run).
"""

from __future__ import annotations

import json
from pathlib import Path

from rubert_clf import data
from rubert_clf.inference import ToxicityClassifier, weights_file

NAME = "rubert"
PROJECT_ROOT = data.PROJECT_ROOT
DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "rubert-toxic"
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_model.py"


class RubertPredictor:
    name = NAME

    def __init__(self, model_dir: Path, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.classifier = ToxicityClassifier.load(self.model_dir, device)
        self.threshold = self.classifier.threshold

    def predict(self, texts):
        probs = self.classifier.predict_proba(list(texts))
        return probs, probs, (probs >= self.threshold).astype(int)

    def describe(self) -> dict:
        selection = self.classifier.selection
        meta_path = self.model_dir / "train_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        weights = weights_file(self.model_dir)
        return {
            "artifact": {"path": data.relative_to(self.model_dir), "weights": weights,
                         "weights_sha256": data.sha256_file(self.model_dir / weights),
                         "base_model": selection.get("base_model"), "epoch": selection["epoch"],
                         "max_length": self.classifier.max_length, "truncation": self.classifier.truncation,
                         "run_report": selection.get("run", {}).get("report")},
            "score": "P(toxic), softmax",
            "probability": "P(toxic), softmax",
            "protocol": {"protocol": selection.get("protocol", "пробное обучение: эпоха и порог на validation"),
                         "epoch_selection": selection.get("selection_rule", selection.get("selection_metric")),
                         "threshold": selection.get("threshold_rule"),
                         "test_used_for_selection": False},
            "training_manifest_sha256": meta.get("data_manifest_sha256"),
        }


def load(artifact: Path | None = None, device: str = "auto") -> RubertPredictor:
    return RubertPredictor(artifact or DEFAULT_ARTIFACT, device)


def prepare(all_models: bool = False, log=print) -> None:
    """Энкодер rubert-tiny2 (тесты, smoke) или все модели протокола — по ревизиям с проверкой хешей."""
    from toxic_core.cli import run_script

    run_script(DOWNLOAD_SCRIPT, ["--all"] if all_models else [])
