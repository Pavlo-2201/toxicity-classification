"""DummyClassifier и word-level TF-IDF + LogisticRegression."""

from toxic_clf.experiments import run_stage

CONFIGS = [
    {"name": "dummy_most_frequent", "model": "dummy", "model_params": {"strategy": "most_frequent"}},
    {"name": "dummy_stratified", "model": "dummy", "model_params": {"strategy": "stratified"}},
    {"name": "word_1_tfidf_logreg", "model": "logreg", "word": {"ngram_range": (1, 1)}},
    # проверка эффекта регистра: тот же baseline без приведения к нижнему регистру
    {"name": "word_1_tfidf_logreg_cased", "model": "logreg", "word": {"ngram_range": (1, 1), "lowercase": False}},
]

if __name__ == "__main__":
    run_stage("stage3_baselines", CONFIGS)
