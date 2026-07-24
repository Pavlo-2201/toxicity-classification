"""Подбор модели и порога без использования test для выбора."""

from __future__ import annotations

from toxic_clf.environment import capture

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from toxic_clf import data, splits
from toxic_clf.experiments import save_results
from toxic_clf.metrics import classification_metrics
from toxic_clf.models import build_pipeline
from toxic_clf.selection import best_f1_threshold, cross_validate_config, out_of_fold_scores, response_method

REPORTS = data.PROJECT_ROOT / "reports"
BEST_CONFIG_PATH = REPORTS / "best_model.json"

FEATURES = {
    "charwb25": {"analyzer": "char_wb", "ngram_range": (2, 5)},
    "char15": {"analyzer": "char", "ngram_range": (1, 5)},
}
MODELS = {"logreg": (4.0, 16.0, 64.0), "linearsvc": (0.3, 1.0, 3.0)}


def candidate_configs() -> list[dict]:
    configs = []
    for feat_name, feat in FEATURES.items():
        for lowercase in (True, False):
            for model, c_values in MODELS.items():
                for c in c_values:
                    configs.append(
                        {
                            "name": f"{feat_name}{'' if lowercase else '_cased'}_{model}_C{c:g}_balanced",
                            "model": model,
                            "model_params": {"C": c, "class_weight": "balanced"},
                            "char": {**feat, "lowercase": lowercase},
                        }
                    )
    return configs


def predict_with_threshold(pipeline, config, texts, threshold):
    method = response_method(config)
    scores = getattr(pipeline, method)(texts)
    scores = scores[:, 1] if method == "predict_proba" else scores
    return (scores >= threshold).astype(int), scores


def save_confusion_matrix(metrics: dict, title: str, path) -> None:
    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    import numpy as np

    disp = ConfusionMatrixDisplay(np.array(matrix), display_labels=["нетоксичный", "токсичный"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set(title=title, xlabel="предсказание", ylabel="истинная метка")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    parts = splits.load_splits()
    train, val, test = parts["train"], parts["validation"], parts["test"]
    x_train, y_train = train[data.TEXT_COL], train[data.LABEL_COL]

    cv_results = []
    for config in candidate_configs():
        cv = cross_validate_config(config, x_train, y_train, n_jobs=args.n_jobs)
        print(f"{config['name']:<45} CV F1tox={cv['f1_toxic']['mean']:.4f}±{cv['f1_toxic']['std']:.4f}")
        cv_results.append({"name": config["name"], "config": config, "split": "train_cv5", "cv": cv})
    cv_results.sort(key=lambda r: r["cv"]["f1_toxic"]["mean"], reverse=True)
    best = cv_results[0]["config"]

    oof = out_of_fold_scores(best, x_train, y_train, n_jobs=args.n_jobs)
    threshold, oof_f1 = best_f1_threshold(y_train, oof)
    default_threshold = 0.5 if response_method(best) == "predict_proba" else 0.0

    pipeline = build_pipeline(best).fit(x_train, y_train)
    val_default, val_scores = predict_with_threshold(pipeline, best, val[data.TEXT_COL], default_threshold)
    val_tuned, _ = predict_with_threshold(pipeline, best, val[data.TEXT_COL], threshold)
    val_metrics_default = classification_metrics(val[data.LABEL_COL], val_default, val_scores)
    val_metrics_tuned = classification_metrics(val[data.LABEL_COL], val_tuned, val_scores)
    use_tuned = val_metrics_tuned["f1_toxic"] >= val_metrics_default["f1_toxic"]
    final_threshold = threshold if use_tuned else default_threshold

    # Test: единственная оценка уже зафиксированной модели и порога
    test_pred, test_scores = predict_with_threshold(pipeline, best, test[data.TEXT_COL], final_threshold)
    test_metrics = classification_metrics(test[data.LABEL_COL], test_pred, test_scores)

    selection = {
        "environment": capture(),
        "selection_metric": "f1_toxic (5-fold CV на train)",
        "best_config": best,
        "response_method": response_method(best),
        "threshold": {
            "default": default_threshold,
            "oof_tuned": round(threshold, 6),
            "oof_f1_toxic": round(oof_f1, 4),
            "used": round(final_threshold, 6),
            "rule": "подобранный порог принимается, если F1 на validation не ниже, чем со стандартным",
            "note": "для LinearSVC порог задан на decision_function и не является вероятностью",
        },
        "validation": {"default_threshold": val_metrics_default, "used_threshold": val_metrics_tuned if use_tuned else val_metrics_default},
        "test": test_metrics,
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
        "manifest_sha256": data.sha256_file(splits.MANIFEST_PATH),
        "content_fingerprint": data.EXPECTED_FINGERPRINT,
    }
    BEST_CONFIG_PATH.write_text(json.dumps(selection, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    save_results("stage6_cv_selection", cv_results)

    save_confusion_matrix(selection["validation"]["used_threshold"], "Validation", REPORTS / "figures" / "confusion_validation.png")
    save_confusion_matrix(test_metrics, "Test", REPORTS / "figures" / "confusion_test.png")

    print(f"Лучшая конфигурация: {best['name']}; порог {final_threshold:.4f} ({'подобранный' if use_tuned else 'стандартный'})")
    print("validation:", json.dumps(selection["validation"]["used_threshold"], ensure_ascii=False))
    print("test:", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
