"""Эксперименты по configs/protocol.json: кандидаты × seed и готовые классификаторы.

    python scripts/experiments.py                      # новый запуск: experiments/runs/<UTC-время>/
    python scripts/experiments.py --out experiments/runs/my --only tiny_L64
    python scripts/experiments.py --resume experiments/runs/<id>   # продолжить прерванный

Новый запуск пишет только в новый пустой каталог: manifest.json (план, отпечаток процедуры,
сессии), splits.json и runs/<кандидат>__s<seed>.json. Продолжение — явный режим: протокол, данные,
ревизии моделей и код должны совпасть с манифестом, пропускаются только прогоны этого же запуска
(src/rubert_clf/experiment_run.py). Проверка готового запуска — scripts/check_experiment_run.py, сводка —
scripts/summarize_experiments.py --run-dir КАТАЛОГ. Опубликованные результаты в experiments/runs скрипт не меняет.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf import data, experiment_run as er, model_source, splits
from rubert_clf.cli import use_utf8_stdio
from rubert_clf.finetune import write_json
from rubert_clf.metrics import best_f1_threshold
from rubert_clf.protocol import RunSpec, calibration_split, evaluate_split, long_mask, run, threshold_free
from rubert_clf.runtime import get_device
from rubert_clf.text import normalize_texts

PROTOCOL = data.PROJECT_ROOT / "configs" / "protocol.json"


def load_parts() -> dict:
    parts = splits.load_splits()
    fit, calib = calibration_split(parts["train"])
    return {"fit": fit, "calibration": calib, "validation": parts["validation"], "test": parts["test"]}


@torch.no_grad()
def ready_scores(spec: dict, texts: list[str], device, batch_size: int = 64) -> np.ndarray:
    model_dir = data.PROJECT_ROOT / spec["dir"]  # не зависит от текущего каталога
    pinned = model_source.verify_registered(model_dir)
    if pinned is None or pinned.revision != spec["revision"]:
        raise ValueError(f"{spec['dir']}: ревизия в протоколе не совпадает с реестром model_source")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
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
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", type=Path, help="каталог нового запуска (должен быть пуст); по умолчанию experiments/runs/<UTC-время>")
    target.add_argument("--resume", type=Path, help="продолжить прерванный запуск из этого каталога")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL, help="протокол нового запуска")
    parser.add_argument("--only", nargs="*", help="новый запуск: только эти кандидаты")
    parser.add_argument("--skip-ready", action="store_true", help="новый запуск: без готовых классификаторов")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.resume and (args.only or args.skip_ready or args.protocol != PROTOCOL):
        parser.error("при --resume план и протокол берутся из manifest.json: --only, --skip-ready и --protocol не нужны")

    run_dir = args.resume or args.out or er.new_run_dir()
    if args.resume:
        if not (run_dir / er.MANIFEST).exists():
            raise SystemExit(f"В {run_dir} нет {er.MANIFEST}: это не каталог запуска")
        previous = er.read_json(run_dir / er.MANIFEST)
        protocol_path, runs = data.PROJECT_ROOT / previous["fingerprint"]["protocol"]["path"], previous["planned"]
    else:
        protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not args.resume:
        try:
            runs = er.plan(protocol, args.only, args.skip_ready)
        except ValueError as error:
            parser.error(str(error))

    device = get_device(args.device)
    parts = load_parts()
    masks = {name: long_mask(parts[name][data.TEXT_COL]) for name in ("validation", "test")}
    try:
        fp = er.fingerprint(protocol_path, protocol, runs, parts, masks)
        env = er.environment(device)
        manifest = er.resume(run_dir, fp, env) if args.resume else er.create(run_dir, fp, env, runs)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for change in er.environment_changes(manifest):
        print(f"ВНИМАНИЕ: окружение отличается от первой сессии — {change}")
    done = er.completed_runs(manifest)
    print(f"{'Продолжение' if args.resume else 'Новый запуск'} {run_dir}: {len(done)} из {len(runs)} прогонов уже есть")
    write_json(run_dir / "splits.json", {
        "fit_rows": len(parts["fit"]), "calibration_rows": len(parts["calibration"]),
        "calibration_source_rows": parts["calibration"][data.ID_COL].tolist(),
        "long_texts": {k: int(v.sum()) for k, v in masks.items()},
    })

    for candidate in protocol["candidates"]:
        for i, seed in enumerate(protocol["seeds"]):
            name = er.candidate_run_name(candidate["name"], seed)
            if name not in runs or name in done:
                continue
            spec = RunSpec(**{**candidate, "model_dir": str(data.PROJECT_ROOT / candidate["model_dir"]), "seed": seed,
                              "extra": {"measure_cpu": i == 0}})
            started = time.perf_counter()
            result = run(spec, parts, device, masks)
            result["wall_seconds"] = round(time.perf_counter() - started, 1)
            er.save_report(run_dir, manifest, name, result)

    for spec in protocol["ready_classifiers"]:
        name = er.ready_run_name(spec)
        if name not in runs or name in done:
            continue
        y = {k: parts[k][data.LABEL_COL].to_numpy() for k in ("calibration", "validation", "test")}
        probs = {k: ready_scores(spec, parts[k][data.TEXT_COL].tolist(), device) for k in y}
        threshold, _ = best_f1_threshold(y["calibration"], probs["calibration"])
        er.save_report(run_dir, manifest, name, {
            "spec": spec, "threshold": float(threshold),
            "validation_threshold_free": threshold_free(y["validation"], probs["validation"]),
            "validation": evaluate_split(y["validation"], probs["validation"], threshold, masks["validation"]),
            "test": evaluate_split(y["test"], probs["test"], threshold, masks["test"]),
            "predictions": {k: np.round(probs[k], 5).tolist() for k in ("validation", "test")},
            "contamination": "обучен в том числе на данных этой задачи или ok.ru — оценка завышена"})
        print(spec["name"], "готово")
    print(f"Запуск {run_dir}: выполнено {len(er.completed_runs(manifest))} из {len(runs)}; "
          f"проверка — python scripts/check_experiment_run.py {run_dir}")


if __name__ == "__main__":
    main()
