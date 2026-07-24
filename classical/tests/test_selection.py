import numpy as np

from toxic_clf.selection import best_f1_threshold, response_method


def test_best_f1_threshold_separates_perfectly():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([-1.2, -0.4, 0.1, 0.3, 2.0])  # decision scores, не вероятности
    threshold, f1 = best_f1_threshold(labels, scores)
    assert f1 == 1.0
    assert ((scores >= threshold).astype(int) == labels).all()


def test_response_method_never_uses_probabilities_for_svc():
    assert response_method({"model": "linearsvc"}) == "decision_function"
    assert response_method({"model": "logreg"}) == "predict_proba"
