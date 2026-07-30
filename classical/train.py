"""Обучение итоговой модели и её сериализация."""

from __future__ import annotations

from toxic_clf.environment import capture

import argparse
import json
from pathlib import Path

from toxic_clf import data, splits
from toxic_clf.final_model import MODEL_PATH, build_final_model, load_selection, save_model, toxicity_scores
from toxic_clf.metrics import bootstrap_f1_ci, classification_metrics

FINAL_METRICS_PATH = data.PROJECT_ROOT / "reports" / "final_metrics.json"
TEST_PREDICTIONS_PATH = data.PROJECT_ROOT / "reports" / "test_predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    selection = load_selection()
    config, threshold = selection["best_config"], selection["threshold"]["used"]
    parts = splits.load_splits()

    model = build_final_model(config, threshold)
    model.fit(parts["train"][data.TEXT_COL], parts["train"][data.LABEL_COL])

    metrics = {}
    for name in ("validation", "test"):
        part = parts[name]
        pred, scores = model.predict(part[data.TEXT_COL]), toxicity_scores(model, part[data.TEXT_COL])
        metrics[name] = classification_metrics(part[data.LABEL_COL], pred, scores)
        if name == "test":
            metrics[name]["f1_toxic_bootstrap_ci"] = bootstrap_f1_ci(part[data.LABEL_COL], pred)
            # Попримерные предсказания для парного сравнения с RuBERT на том же test
            part[[data.ID_COL, data.LABEL_COL]].assign(score=scores.round(6), pred=pred).to_csv(
                TEST_PREDICTIONS_PATH, index=False
            )
        print(f"{name}: " + json.dumps({k: v for k, v in metrics[name].items()}, ensure_ascii=False))

    metadata = {
        "environment": capture(),
        "config": config,
        "threshold": threshold,
        "score_type": "decision_function (не вероятность)" if config["model"] == "linearsvc" else "вероятность",
        "trained_on": "train",
        "train_rows": int(len(parts["train"])),
        "manifest_sha256": data.sha256_file(splits.MANIFEST_PATH),
        "content_fingerprint": data.EXPECTED_FINGERPRINT,
        "metrics": metrics,
    }
    save_model(model, metadata, args.output)
    FINAL_METRICS_PATH.write_text(
        json.dumps({"environment": capture(), "model": config["name"], "threshold": threshold, **metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Модель сохранена: {args.output}")


if __name__ == "__main__":
    main()
