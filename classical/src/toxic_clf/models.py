"""Сборка полного sklearn Pipeline из описания конфигурации."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer

from toxic_clf.text import normalize_texts, strip_punctuation_texts

SEED = 42

WORD_DEFAULTS = {
    "analyzer": "word",
    "lowercase": True,
    "ngram_range": (1, 1),
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"(?u)\b\w+\b",  # оставляем и однобуквенные слова («я», «в»)
}

CHAR_DEFAULTS = {
    "analyzer": "char_wb",
    "lowercase": True,
    "ngram_range": (2, 5),
    "min_df": 2,
    "sublinear_tf": True,
}


def _tfidf(defaults: dict, overrides: dict) -> TfidfVectorizer:
    params = {**defaults, **overrides}
    params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def build_features(config: dict):
    blocks = []
    if "word" in config or "char" not in config:
        blocks.append(("word", _tfidf(WORD_DEFAULTS, config.get("word", {}))))
    if "char" in config:
        blocks.append(("char", _tfidf(CHAR_DEFAULTS, config["char"])))
    if len(blocks) == 1:
        return blocks[0][1]
    return FeatureUnion(blocks)


def build_classifier(config: dict):
    model = config["model"]
    params = config.get("model_params", {})
    if model == "dummy":
        return DummyClassifier(**{"strategy": "most_frequent", "random_state": SEED, **params})
    if model == "logreg":
        return LogisticRegression(**{"C": 1.0, "max_iter": 2000, "solver": "liblinear", "random_state": SEED, **params})
    if model == "linearsvc":
        # LinearSVC даёт только decision_function; это не вероятности
        return LinearSVC(**{"C": 1.0, "max_iter": 5000, "random_state": SEED, **params})
    raise ValueError(f"Неизвестная модель: {model}")


def build_pipeline(config: dict) -> Pipeline:
    steps = [("normalize", FunctionTransformer(normalize_texts))]
    if config.get("strip_punctuation"):
        steps.append(("strip_punctuation", FunctionTransformer(strip_punctuation_texts)))
    steps += [("features", build_features(config)), ("clf", build_classifier(config))]
    return Pipeline(steps)


def toxicity_scores(pipeline: Pipeline, texts):
    """Оценки токсичности для ранжирования: вероятность, если модель её даёт, иначе decision_function."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)
