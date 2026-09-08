"""Эксперименты по configs/protocol.json: кандидаты × seed и готовые классификаторы.

Каждый запуск пишет experiments/runs/runs/<кандидат>__s<seed>.json; уже готовые запуски пропускаются,
поэтому прерванный прогон можно продолжить той же командой.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf import data, splits
from rubert_clf.cli import use_utf8_stdio
from rubert_clf.finetune import write_json
from rubert_clf.metrics import best_f1_threshold
from rubert_clf.protocol import RunSpec, calibration_split, evaluate_split, long_mask, run, threshold_free
from rubert_clf.runtime import get_device
from rubert_clf.text import normalize_texts

PROTOCOL = data.PROJECT_ROOT / "configs" / "protocol.json"
OUT = data.PROJECT_ROOT / "experiments" / "runs"


def load_parts() -> dict:
    parts = splits.load_splits()
    fit, calib = calibration_split(parts["train"])
    return {"fit": fit, "calibration": calib, "validation": parts["validation"], "test": parts["test"]}


@torch.no_grad()
def ready_scores(spec: dict, texts: list[str], device, batch_size: int = 64) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(spec["dir"])
    model = AutoModelForSequenceClassification.from_pretrained(spec["dir"]).to(device).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    texts = normalize_texts(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.empty(len(texts))
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        if "non-toxic" in labels:  # мультиметочная голова cointegrated: сигмоида по классам
            out[idx] = 1 - torch.sigmoid(logits)[:, labels["non-toxic"]].cpu().numpy()
        else:
            out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="запустить только этих кандидатов")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-ready", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    device = get_device(args.device)
    parts = load_parts()
    masks = {name: long_mask(parts[name][data.TEXT_COL]) for name in ("validation", "test")}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "splits.json", {
        "fit_rows": len(parts["fit"]), "calibration_rows": len(parts["calibration"]),
        "calibration_source_rows": parts["calibration"][data.ID_COL].tolist(),
        "long_texts": {k: int(v.sum()) for k, v in masks.items()}, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })

    for candidate in protocol["candidates"]:
        if args.only and candidate["name"] not in args.only:
            continue
        for i, seed in enumerate(protocol["seeds"]):
            path = OUT / "runs" / f"{candidate['name']}__s{seed}.json"
            if path.exists():
                continue
            spec = RunSpec(**{**candidate, "model_dir": str(data.PROJECT_ROOT / candidate["model_dir"]), "seed": seed,
                              "extra": {"measure_cpu": i == 0}})
            started = time.perf_counter()
            result = run(spec, parts, device, masks)
            result["wall_seconds"] = round(time.perf_counter() - started, 1)
            write_json(path, result)

    if not args.skip_ready:
        for spec in protocol["ready_classifiers"]:
            path = OUT / "runs" / f"ready__{spec['name'].split('/')[-1]}.json"
            if path.exists():
                continue
            y = {k: parts[k][data.LABEL_COL].to_numpy() for k in ("calibration", "validation", "test")}
            probs = {k: ready_scores(spec, parts[k][data.TEXT_COL].tolist(), device) for k in y}
            threshold, _ = best_f1_threshold(y["calibration"], probs["calibration"])
            write_json(path, {"spec": spec, "threshold": float(threshold),
                              "validation_threshold_free": threshold_free(y["validation"], probs["validation"]),
                              "validation": evaluate_split(y["validation"], probs["validation"], threshold, masks["validation"]),
                              "test": evaluate_split(y["test"], probs["test"], threshold, masks["test"]),
                              "predictions": {k: np.round(probs[k], 5).tolist() for k in ("validation", "test")},
                              "contamination": "обучен в том числе на данных этой задачи или ok.ru — оценка завышена"})
            print(spec["name"], "готово")


if __name__ == "__main__":
    main()
