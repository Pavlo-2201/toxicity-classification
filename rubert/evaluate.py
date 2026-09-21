"""Оценка сохранённой модели на разбиении с порогом из selection.json."""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json
import time
from pathlib import Path

from rubert_clf import data
from rubert_clf.finetune import write_json
from rubert_clf.inference import ToxicityClassifier
from toxic_core import splits
from toxic_core.metrics import bootstrap_f1_ci, classification_metrics
from toxic_core.results import OutputExistsError, ensure_absent
from toxic_core.stdio import use_utf8_stdio


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=data.PROJECT_ROOT / "models" / "rubert-toxic")
    parser.add_argument("--split", choices=splits.SPLITS, default="validation")
    parser.add_argument("--limit", type=int, help="оценить на первых N текстах разбиения (для проверки)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="заменить существующий отчёт")
    args = parser.parse_args()
    out = data.PROJECT_ROOT / "reports" / "eval" / f"{args.model_dir.name}_{args.split}{'_limit' + str(args.limit) if args.limit else ''}.json"
    try:
        ensure_absent([out], args.overwrite)
    except OutputExistsError as error:
        raise SystemExit(str(error)) from error

    clf = ToxicityClassifier.load(args.model_dir, args.device)
    df = splits.load_splits()[args.split]
    if args.limit:
        df = df.head(args.limit)
    y = df[data.LABEL_COL].values
    start = time.perf_counter()
    probs = clf.predict_proba(df[data.TEXT_COL])
    seconds = time.perf_counter() - start
    pred = (probs >= clf.threshold).astype(int)
    report = {
        "environment": capture(neural=True),
        "model_dir": args.model_dir.name,
        "split": args.split,
        "n": len(df),
        "threshold": clf.threshold,
        "selected_epoch": clf.selection["epoch"],
        "device": str(clf.device),
        "inference_seconds": round(seconds, 2),
        "metrics": classification_metrics(y, pred, probs),
        "f1_bootstrap_ci": bootstrap_f1_ci(y, pred),
    }
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
