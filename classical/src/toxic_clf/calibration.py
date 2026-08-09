"""Вероятности для LinearSVC: Platt-шкалирование на out-of-fold margin обучающей выборки.

`decision_function` LinearSVC — это margin (расстояние до разделяющей гиперплоскости),
а не вероятность; в отчётах он остаётся столбцом `score`. Вероятность получается
логистической функцией от margin, параметры которой подобраны на OOF-оценках train
(5-fold): ни validation, ни test в калибровке не участвуют.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, margins, labels) -> "PlattCalibrator":
        # почти без регуляризации: два параметра на тысячах точек
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(np.asarray(margins).reshape(-1, 1), labels)
        return cls(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))

    def predict_proba(self, margins) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.slope * np.asarray(margins, dtype=float) + self.intercept)))

    def to_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(labels, probs, n_bins: int = 10) -> float:
    """ECE по равным интервалам вероятности: средняя |доля токсичных − средняя вероятность|."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(total)


def calibration_metrics(labels, probs) -> dict:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier": round(float(brier_score_loss(labels, probs)), 4),
        "log_loss": round(float(log_loss(labels, probs)), 4),
        "ece_10_bins": round(expected_calibration_error(labels, probs), 4),
    }
