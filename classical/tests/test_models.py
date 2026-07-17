import numpy as np

from toxic_clf.metrics import classification_metrics
from toxic_clf.models import build_pipeline
from toxic_clf.text import normalize_text

TOXIC = ["ты тупой идиот", "заткнись, дебил!!!", "иди нафиг, урод", "какой же ты идиот"]
CLEAN = ["спасибо за статью", "хороший день сегодня", "интересная мысль, согласен", "поддерживаю автора"]
TEXTS = TOXIC * 3 + CLEAN * 3
LABELS = [1] * 12 + [0] * 12


def test_normalize_keeps_case_and_punctuation():
    assert normalize_text("  Ну ТЫ и...   урод!!!\n") == "Ну ТЫ и... урод!!!"
    assert normalize_text("é") == "é"  # NFC


def test_metrics_confusion_matrix_and_f1():
    m = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0], scores=[0.9, 0.2, 0.6, 0.1])
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["f1_toxic"] == 0.5
    assert m["roc_auc"] == 0.75


def test_logreg_pipeline_learns_toy_data():
    pipe = build_pipeline({"name": "t", "model": "logreg", "word": {"min_df": 1}})
    pipe.fit(TEXTS, LABELS)
    assert (pipe.predict(TEXTS) == np.array(LABELS)).all()


def test_vectorizer_is_fitted_on_training_texts_only():
    pipe = build_pipeline({"name": "t", "model": "logreg", "word": {"min_df": 1}})
    pipe.fit(TEXTS, LABELS)
    vocab = set(pipe.named_steps["features"].vocabulary_)
    pipe.predict(["совершенно новое словечко"])
    assert "словечко" not in vocab
    assert set(pipe.named_steps["features"].vocabulary_) == vocab


def test_linearsvc_pipeline_exposes_decision_scores_not_probabilities():
    from toxic_clf.models import toxicity_scores

    pipe = build_pipeline({"name": "t", "model": "linearsvc", "word": {"min_df": 1, "ngram_range": (1, 2)}})
    pipe.fit(TEXTS, LABELS)
    assert not hasattr(pipe, "predict_proba")
    scores = toxicity_scores(pipe, TEXTS)
    np.testing.assert_allclose(scores, pipe.decision_function(TEXTS))
    assert ((scores > 0).astype(int) == pipe.predict(TEXTS)).all()


def test_word_char_union_and_punctuation_ablation():
    from toxic_clf.text import strip_punctuation_texts

    pipe = build_pipeline(
        {"name": "t", "model": "logreg", "word": {"min_df": 1}, "char": {"min_df": 1}, "strip_punctuation": True}
    )
    pipe.fit(TEXTS, LABELS)
    names = pipe.named_steps["features"].get_feature_names_out()
    assert any(n.startswith("word__") for n in names) and any(n.startswith("char__") for n in names)
    assert not any("!" in n for n in names)
    assert strip_punctuation_texts(["Ну, ты!!! :)"]) == ["Ну ты"]
