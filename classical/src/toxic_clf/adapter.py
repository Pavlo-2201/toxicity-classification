"""Классическая модель в общем интерфейсе (toxic_core.registry).

score — margin LinearSVC (decision_function), к нему применяется порог артефакта; probability —
Platt-калибровка на OOF margin train. Протокол: конфигурация выбрана 5-fold CV на train, порог —
F1-оптимум по OOF-оценкам train (принимается, если на validation F1 не ниже, чем при пороге по
умолчанию). Test в выборе не участвует.
"""

from __future__ import annotations

from pathlib import Path

from toxic_clf import data
from toxic_clf.final_model import MODEL_PATH, load_calibrator, load_metadata, load_model, toxicity_scores

NAME = "classical"
PROJECT_ROOT = data.PROJECT_ROOT
DEFAULT_ARTIFACT = MODEL_PATH
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

PROTOCOL = {
    "model_selection": "5-fold CV на train (classical/reports/best_model.json)",
    "threshold": "F1-оптимум по OOF-оценкам train (5-fold); принимается, если F1 на validation не ниже, "
                 "чем при пороге по умолчанию",
    "test_used_for_selection": False,
}


class ClassicalPredictor:
    name = NAME

    def __init__(self, path: Path):
        self.path = Path(path)
        self.model = load_model(self.path)
        self.calibrator = load_calibrator(self.path)
        self.metadata = load_metadata(self.path)
        self.threshold = float(self.model.threshold)

    def predict(self, texts):
        texts = list(texts)
        scores = toxicity_scores(self.model, texts)
        probabilities = self.calibrator.predict_proba(scores) if self.calibrator else None
        return scores, probabilities, self.model.predict(texts)

    def describe(self) -> dict:
        # без SHA-256 joblib: байты pickle зависят от платформы, а отчёт должен воспроизводиться в CI на любой ОС
        meta = self.metadata
        return {
            "artifact": {"path": data.relative_to(self.path), "config": meta.get("config", {}).get("name"),
                         "normalization": meta.get("normalization", "baseline"), "sklearn": meta.get("sklearn_version")},
            "score": "decision_function LinearSVC (margin, не вероятность)" if self.calibrator else "predict_proba",
            "probability": "Platt на OOF margin train (5-fold)" if self.calibrator else None,
            "protocol": {**PROTOCOL, "threshold_selection": meta.get("threshold_selection")},
            "training_manifest_sha256": meta.get("manifest_sha256"),
        }


def load(artifact: Path | None = None, device: str | None = None) -> ClassicalPredictor:
    return ClassicalPredictor(artifact or DEFAULT_ARTIFACT)


def prepare(all_models: bool = False, log=print) -> None:
    log("classical: кроме данных ничего не нужно")
