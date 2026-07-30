"""Итоговая модель: полный sklearn Pipeline с зафиксированным порогом и его сериализация."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import FixedThresholdClassifier

from toxic_clf import data
from toxic_clf.models import build_pipeline
from toxic_clf.selection import response_method

MODELS_DIR = data.PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "toxic_clf.joblib"
BEST_MODEL_PATH = data.PROJECT_ROOT / "reports" / "best_model.json"


def load_selection(path: Path = BEST_MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_final_model(config: dict, threshold: float) -> FixedThresholdClassifier:
    """Нормализация -> TF-IDF -> классификатор, обёрнутые порогом.

    Для LinearSVC порог применяется к decision_function; predict_proba у модели нет.
    """
    return FixedThresholdClassifier(build_pipeline(config), threshold=threshold, response_method=response_method(config))


def toxicity_scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def save_model(model, metadata: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {**metadata, "sklearn_version": sklearn.__version__, "model_file": path.name}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден. Запустите: python train.py")
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sklearn_version") != sklearn.__version__:
            print(f"Внимание: модель сохранена в scikit-learn {meta.get('sklearn_version')}, установлена {sklearn.__version__}")
    return joblib.load(path)
