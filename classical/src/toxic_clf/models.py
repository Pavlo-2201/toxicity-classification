"""Сборка полного sklearn Pipeline из описания конфигурации."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from toxic_clf.text import normalize_texts

SEED = 42

WORD_DEFAULTS = {
    "analyzer": "word",
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"(?u)\b\w+\b",  # оставляем и однобуквенные слова («я», «в»)
}


def build_vectorizer(config: dict) -> TfidfVectorizer:
    params = {**WORD_DEFAULTS, **config.get("word", {})}
    params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def build_classifier(config: dict):
    model = config["model"]
    params = config.get("model_params", {})
    if model == "dummy":
        return DummyClassifier(**{"strategy": "most_frequent", "random_state": SEED, **params})
    if model == "logreg":
        return LogisticRegression(**{"C": 1.0, "max_iter": 2000, "solver": "liblinear", "random_state": SEED, **params})
    raise ValueError(f"Неизвестная модель: {model}")


def build_pipeline(config: dict) -> Pipeline:
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_texts)),
            ("features", build_vectorizer(config)),
            ("clf", build_classifier(config)),
        ]
    )


def toxicity_scores(pipeline: Pipeline, texts):
    """Оценки токсичности для ранжирования: вероятность, если модель её даёт, иначе decision_function."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)
