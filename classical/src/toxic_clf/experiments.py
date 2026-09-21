"""Запуск конфигураций: обучение на train, оценка на validation, сохранение результатов."""

from __future__ import annotations

from toxic_core.environment import capture

import json
import time
from pathlib import Path


from toxic_clf import data
from toxic_clf.models import build_pipeline, toxicity_scores
from toxic_core import splits
from toxic_core.metrics import classification_metrics

EXPERIMENTS_DIR = data.PROJECT_ROOT / "reports" / "experiments"


def evaluate_config(config: dict, train, validation) -> dict:
    pipeline = build_pipeline(config)
    start = time.perf_counter()
    pipeline.fit(train[data.TEXT_COL], train[data.LABEL_COL])
    fit_seconds = time.perf_counter() - start
    y_pred = pipeline.predict(validation[data.TEXT_COL])
    scores = toxicity_scores(pipeline, validation[data.TEXT_COL])
    n_features = len(pipeline.named_steps["features"].get_feature_names_out())
    return {
        "name": config["name"],
        "config": config,
        "split": "validation",
        "metrics": classification_metrics(validation[data.LABEL_COL], y_pred, scores),
        "n_features": n_features,
        "fit_seconds": round(fit_seconds, 2),
    }


def run_stage(stage: str, configs: list[dict]) -> list[dict]:
    parts = splits.load_splits()
    results = []
    for config in configs:
        result = evaluate_config(config, parts["train"], parts["validation"])
        m = result["metrics"]
        print(f"{config['name']:<40} F1tox={m['f1_toxic']:.4f} macroF1={m['macro_f1']:.4f} "
              f"P={m['precision_toxic']:.4f} R={m['recall_toxic']:.4f}")
        results.append(result)
    save_results(stage, results)
    return results


def save_results(stage: str, results: list[dict], path: Path | None = None) -> Path:
    path = path or EXPERIMENTS_DIR / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "environment": capture(),
        "manifest_sha256": data.sha256_file(splits.MANIFEST_PATH),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8", newline="\n")
    return path
