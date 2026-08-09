"""Внешняя оценка итоговой модели на ok.ru после проверки пересечений с нашим корпусом.

Модель обучена только на train нашего корпуса; порог и калибровка — те же, что в
итоговой модели. Пишет reports/external_okru.json и попримерные предсказания без текстов
(reports/external_okru_predictions.csv: номер строки источника, метка, score, решение,
вероятность).
"""

from __future__ import annotations

from toxic_clf.environment import capture

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from toxic_clf import data
from toxic_clf.calibration import calibration_metrics
from toxic_clf.cli import use_utf8_stdio
from toxic_clf.external import OKRU, clean_external, download_okru, load_okru
from toxic_clf.final_model import MODEL_PATH, load_calibrator, load_model, toxicity_scores
from toxic_clf.metrics import bootstrap_f1_ci, classification_metrics

REPORTS = data.PROJECT_ROOT / "reports"


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    download_okru()
    okru, overlap = clean_external(load_okru(), data.load_dataset())
    model, calibrator = load_model(args.model), load_calibrator(args.model)
    texts, y = okru["text"].tolist(), okru["label"].to_numpy()
    scores = toxicity_scores(model, texts)
    pred = model.predict(texts)
    probs = calibrator.predict_proba(scores) if calibrator else None
    metrics = classification_metrics(y, pred, scores)
    metrics["f1_toxic_bootstrap_ci"] = bootstrap_f1_ci(y, pred)
    if probs is not None:
        metrics["calibration_oof_platt"] = calibration_metrics(y, probs)
        metrics["mean_probability"] = round(float(probs.mean()), 4)
    # порог, лучший для ok.ru, — только как ориентир, насколько дело в сдвиге порога
    order = np.sort(np.unique(scores))
    best = max(((float(t), float(classification_metrics(y, (scores >= t).astype(int))["f1_toxic"]))
                for t in np.quantile(order, np.linspace(0.01, 0.99, 197))), key=lambda x: x[1])
    report = {
        "environment": capture(),
        "dataset": OKRU, "overlap_check": overlap,
        "model": str(args.model), "metrics_at_model_threshold": metrics,
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "average_precision": round(float(average_precision_score(y, scores)), 4),
        "oracle_threshold_on_okru": {"threshold": round(best[0], 4), "f1_toxic": round(best[1], 4),
                                     "note": "подобран на самих данных ok.ru: верхняя граница, не результат"},
    }
    (REPORTS / "external_okru.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    okru[["okru_row", "label"]].assign(score=np.round(scores, 6), pred=pred,
                                       **({"probability": np.round(probs, 6)} if probs is not None else {})).to_csv(
        REPORTS / "external_okru_predictions.csv", index=False, lineterminator="\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
