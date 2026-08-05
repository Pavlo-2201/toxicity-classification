import numpy as np

from toxic_clf.calibration import PlattCalibrator, expected_calibration_error
from toxic_clf.robustness import NORMALIZATIONS, PERTURBATIONS, collapse_repeats, map_homoglyphs, perturb, remove_inner_masks

TEXT = "Ну ты и дурачок, сколько можно писать одно и то же в этой ветке обсуждения"


def test_perturbations_are_deterministic_and_change_text():
    for kind in [*PERTURBATIONS, "combo"]:
        assert perturb(TEXT, kind, 1) == perturb(TEXT, kind, 1)
        assert perturb(TEXT, kind, 1) != TEXT or kind == "typo"
    assert perturb(TEXT, "combo", 1) != TEXT


def test_normalizations_undo_their_own_distortion_only():
    assert map_homoglyphs("сooбщение и John") == "сообщение и John"  # латиница без кириллицы не трогается
    assert collapse_repeats("ну ооочень хорошо") == "ну очень хорошо"  # двойные буквы остаются
    assert remove_inner_masks("ну ты д*бил") == "ну ты дбил"
    assert remove_inner_masks("т.е. так") == "те. так"  # цена варианта на обычном тексте
    assert NORMALIZATIONS["all"](perturb(TEXT, "repeat", 3)) == NORMALIZATIONS["all"](TEXT)
    assert NORMALIZATIONS["all"](perturb(TEXT, "homoglyph", 3)) == NORMALIZATIONS["all"](TEXT)


def test_platt_is_monotone_and_calibrates_logistic_data():
    rng = np.random.default_rng(0)
    margins = rng.normal(0, 2, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-(1.5 * margins - 0.5)))).astype(int)
    calibrator = PlattCalibrator.fit(margins, labels)
    assert abs(calibrator.slope - 1.5) < 0.1 and abs(calibrator.intercept + 0.5) < 0.1
    probs = calibrator.predict_proba(np.sort(margins))
    assert np.all(np.diff(probs) >= 0)
    assert expected_calibration_error(labels, calibrator.predict_proba(margins)) < 0.02
    assert expected_calibration_error(labels, np.full(20000, 0.99)) > 0.4
