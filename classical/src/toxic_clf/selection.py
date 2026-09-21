"""Подбор конфигурации кросс-валидацией на train и выбор порога по out-of-fold оценкам."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from toxic_clf.models import SEED, build_pipeline
from toxic_core.metrics import best_f1_threshold, classification_metrics

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


THRESHOLD_RULE = "подобранный порог принимается, если F1 на validation не ниже, чем со стандартным"


def choose_threshold(config: dict, oof_scores, y_train, val_scores, y_val) -> dict:
    """Порог по правилу scripts/select_model.py: F1-оптимальный на OOF train принимается,
    если на validation F1 (с округлением отчёта) не ниже, чем со стандартным порогом."""
    tuned, oof_f1 = best_f1_threshold(y_train, oof_scores)
    default = 0.5 if response_method(config) == "predict_proba" else 0.0
    val_scores = np.asarray(val_scores)
    f1 = {t: classification_metrics(y_val, (val_scores >= t).astype(int))["f1_toxic"] for t in (default, tuned)}
    used = tuned if f1[tuned] >= f1[default] else default
    return {"default": default, "oof_tuned": round(tuned, 6), "oof_f1_toxic": round(oof_f1, 4),
            "used": round(used, 6), "rule": THRESHOLD_RULE}

