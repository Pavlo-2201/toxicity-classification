import numpy as np

from toxic_clf.final_model import build_final_model, load_model, save_model, toxicity_scores

TOXIC = ["ты тупой идиот", "заткнись, дебил!!!", "иди нафиг, урод", "какой же ты идиот"]
CLEAN = ["спасибо за статью", "хороший день сегодня", "интересная мысль, согласен", "поддерживаю автора"]
TEXTS = TOXIC * 3 + CLEAN * 3
LABELS = [1] * 12 + [0] * 12
CONFIG = {
    "name": "t",
    "model": "linearsvc",
    "model_params": {"C": 0.3, "class_weight": "balanced"},
    "char": {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 1},
}
NEW = ["ну ты и идиот", "отличная статья, спасибо", "  Заткнись!!!\n"]


def test_threshold_is_applied_to_decision_scores():
    model = build_final_model(CONFIG, threshold=0.25).fit(TEXTS, LABELS)
    scores = toxicity_scores(model, TEXTS + NEW)
    assert not hasattr(model, "predict_proba")
    assert (model.predict(TEXTS + NEW) == (scores >= 0.25).astype(int)).all()


def test_save_load_roundtrip_gives_identical_predictions(tmp_path):
    model = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    path = tmp_path / "model.joblib"
    save_model(model, {"config": CONFIG}, path)
    loaded = load_model(path)
    assert (path.with_suffix(".json")).exists()
    np.testing.assert_array_equal(loaded.predict(NEW), model.predict(NEW))
    np.testing.assert_allclose(toxicity_scores(loaded, NEW), toxicity_scores(model, NEW))


def test_training_is_reproducible():
    a = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    b = build_final_model(CONFIG, threshold=0.0).fit(TEXTS, LABELS)
    np.testing.assert_allclose(toxicity_scores(a, NEW), toxicity_scores(b, NEW))
