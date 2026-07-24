"""Подбор конфигурации кросс-валидацией на train и выбор порога по out-of-fold оценкам."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from toxic_clf.models import SEED, build_pipeline

N_FOLDS = 5


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cross_validate_config(config: dict, texts, labels, n_jobs: int = -1) -> dict:
    scores = cross_validate(
        build_pipeline(config), texts, labels, cv=cv_splitter(), n_jobs=n_jobs,
        scoring={"f1_toxic": "f1", "macro_f1": "f1_macro", "precision_toxic": "precision", "recall_toxic": "recall"},
    )
    out = {}
    for key in ("f1_toxic", "macro_f1", "precision_toxic", "recall_toxic"):
        values = scores[f"test_{key}"]
        out[key] = {"mean": round(float(values.mean()), 4), "std": round(float(values.std()), 4)}
    return out


def response_method(config: dict) -> str:
    """LinearSVC не даёт вероятностей: порог задаётся на decision_function."""
    return "predict_proba" if config["model"] == "logreg" else "decision_function"


def out_of_fold_scores(config: dict, texts, labels, n_jobs: int = -1) -> np.ndarray:
    method = response_method(config)
    scores = cross_val_predict(build_pipeline(config), texts, labels, cv=cv_splitter(), method=method, n_jobs=n_jobs)
    return scores[:, 1] if method == "predict_proba" else scores


def best_f1_threshold(labels, scores) -> tuple[float, float]:
    """Порог, максимизирующий F1 токсичного класса. Возвращает (порог, F1)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
