"""Обучение итоговой модели и её сериализация.

Конфигурация берётся из reports/best_model.json. С --normalization <вариант> обучается
та же конфигурация с другой нормализацией (docs/robustness.md): порог и калибровка
подбираются заново на OOF этого варианта, артефакт и отчёты пишутся отдельно и
baseline не перезаписывают.

Test в обучении не участвует: fit_final видит только train (OOF для порога и калибровки) и
validation (проверка порога). Готовые артефакт и отчёты не перезаписываются без --overwrite.
"""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json

import numpy as np
from pathlib import Path

from toxic_clf import data
from toxic_clf.calibration import PlattCalibrator, calibration_metrics
from toxic_core import splits
from toxic_clf.final_model import (build_final_model, load_selection, model_path_for, reports_dir_for, save_model,
                                   toxicity_scores)
from toxic_clf.robustness import NORMALIZATIONS
from toxic_clf.selection import choose_threshold, out_of_fold_scores
from toxic_core.metrics import bootstrap_f1_ci, classification_metrics
from toxic_core.results import OutputExistsError, ensure_absent
from toxic_core.stdio import use_utf8_stdio


def fit_final(config: dict, train, validation):
    """Модель, калибратор и выбор порога только по train (OOF) и validation."""
    x_train, y_train = train[data.TEXT_COL], train[data.LABEL_COL]
    # OOF-оценки train (5-fold): по ним выбирается порог и обучается калибровка
    oof = out_of_fold_scores(config, x_train, y_train)
    model = build_final_model(config, threshold=0.0).fit(x_train, y_train)
    threshold_info = choose_threshold(config, oof, y_train, toxicity_scores(model, validation[data.TEXT_COL]),
                                      validation[data.LABEL_COL])
    model.set_params(threshold=threshold_info["used"])
    # вероятность: Platt-шкалирование на OOF margin train; score остаётся сырым margin
    calibrator = PlattCalibrator.fit(oof, y_train) if config["model"] == "linearsvc" else None
    return model, calibrator, threshold_info


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalization", choices=sorted(NORMALIZATIONS), default="baseline",
                        help="вариант нормализации; не baseline — отдельный артефакт со своим порогом и калибровкой")
    parser.add_argument("--output", type=Path, help="артефакт; по умолчанию models/toxic_clf[_<вариант>].joblib")
    parser.add_argument("--reports-dir", type=Path,
                        help="куда писать final_metrics.json и test_predictions.csv; по умолчанию reports/ "
                             "для baseline и reports/normalization_<вариант>/ для варианта")
    parser.add_argument("--overwrite", action="store_true", help="заменить существующие артефакт и отчёты")
    args = parser.parse_args()
    output = args.output or model_path_for(args.normalization)
    reports_dir = args.reports_dir or reports_dir_for(args.normalization)
    try:
        ensure_absent([output, output.with_suffix(".json"), reports_dir / "final_metrics.json",
                       reports_dir / "test_predictions.csv"], args.overwrite, hint="--output и --reports-dir")
    except OutputExistsError as error:
        raise SystemExit(str(error)) from error

    selection = load_selection()
    config = dict(selection["best_config"])
    if args.normalization != "baseline":
        config.update(name=f"{config['name']}|{args.normalization}", normalization=args.normalization)
    parts = splits.load_splits()
    model, calibrator, threshold_info = fit_final(config, parts["train"], parts["validation"])
    threshold = threshold_info["used"]
    if args.normalization == "baseline" and threshold != selection["threshold"]["used"]:
        raise SystemExit(f"Порог {threshold} не совпал с reports/best_model.json ({selection['threshold']['used']}): "
                         "выбор модели не воспроизводится")

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in ("validation", "test"):
        part = parts[name]
        pred, scores = model.predict(part[data.TEXT_COL]), toxicity_scores(model, part[data.TEXT_COL])
        metrics[name] = classification_metrics(part[data.LABEL_COL], pred, scores)
        probs = calibrator.predict_proba(scores) if calibrator else scores
        if calibrator:
            metrics[name]["calibration_oof_platt"] = calibration_metrics(part[data.LABEL_COL], probs)
            # для сравнения: логистическая функция от сырого margin без подбора параметров
            metrics[name]["calibration_raw_sigmoid"] = calibration_metrics(part[data.LABEL_COL], 1 / (1 + np.exp(-scores)))
        if name == "test":
            metrics[name]["f1_toxic_bootstrap_ci"] = bootstrap_f1_ci(part[data.LABEL_COL], pred)
            # Попримерные предсказания для парного сравнения с RuBERT на том же test
            # score — сырой margin LinearSVC (не вероятность), probability — OOF Platt
            part[[data.ID_COL, data.LABEL_COL]].assign(score=scores.round(6), pred=pred,
                                                       probability=np.round(probs, 6)).to_csv(
                reports_dir / "test_predictions.csv", index=False, lineterminator="\n"
            )
        print(f"{name}: " + json.dumps({k: v for k, v in metrics[name].items()}, ensure_ascii=False))

    metadata = {
        "environment": capture(),
        "config": config,
        "normalization": args.normalization,
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "score_type": "decision_function (не вероятность)" if config["model"] == "linearsvc" else "вероятность",
        "calibration": {"method": "Platt на OOF margin train (5-fold)", **calibrator.to_dict()} if calibrator else None,
        "trained_on": "train",
        "train_rows": int(len(parts["train"])),
        "manifest_sha256": data.sha256_file(splits.MANIFEST_PATH),
        "content_fingerprint": data.EXPECTED_FINGERPRINT,
        "metrics": metrics,
    }
    save_model(model, metadata, output)
    (reports_dir / "final_metrics.json").write_text(
        json.dumps({"environment": capture(), "model": config["name"], "normalization": args.normalization, "threshold": threshold,
                    "threshold_selection": threshold_info, **metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Модель сохранена: {output}; отчёты: {reports_dir}")


if __name__ == "__main__":
    main()
