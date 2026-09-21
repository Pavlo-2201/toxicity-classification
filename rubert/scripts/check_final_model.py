"""Проверка финального пути после сериализации: артефакт → предсказания → порог → отчёт.

    python scripts/check_final_model.py                  # models/rubert-toxic, validation и test
    python scripts/check_final_model.py --device cpu
    python scripts/check_final_model.py --model-dir models/rubert-toxic-lora   # LoRA-адаптер

1. Загрузка через ToxicityClassifier.load: веса (SHA-256), эпоха и порог сверяются с отчётом прогона.
2. Токенизатор артефакта даёт те же input_ids, что токенизатор базовой модели из реестра.
3. Условия прогона (тот же батч, на GPU — autocast float16): вероятности validation и test совпадают
   с вероятностями отчёта прогона (там они округлены до 5 знаков). Без GPU пункт пропускается.
   LoRA здесь — с адаптером, без слияния, как при обучении.
4. Путь predict.py и evaluate.py (float32, LoRA — слитая с весами): решения при сохранённом пороге совпадают с отчётом прогона;
   отличаться могут только тексты, чья вероятность ближе к порогу, чем расхождение вероятностей.
   Матрицы ошибок — как в отчёте прогона и в reports/eval/rubert-toxic_<split>.json.
5. CLI predict.py на текстах test (ближайших к порогу и первых) выдаёт те же вероятности и решения.

Итог — reports/eval/<модель>_check.json (без текстов, только source_row); при расхождении — код выхода 1.
"""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from rubert_clf import data, model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.finetune import predict_proba, write_json
from rubert_clf.inference import ToxicityClassifier, is_peft_model, weights_file
from toxic_core import splits
from toxic_core.metrics import classification_metrics
from toxic_core.results import OutputExistsError, ensure_absent
from toxic_core.stdio import use_utf8_stdio
from toxic_core.text import normalize_texts

RUN_ATOL = 1e-4  # вероятности отчёта округлены до 5 знаков
FLOAT32_ATOL = 1e-2  # float32 против float16 прогона: заведомо больше ожидаемых ~1e-3
CLI_ATOL = 1e-4  # predict.py печатает 4 знака
CLI_TEXTS = 8


def tokenizer_check(clf: ToxicityClassifier, texts: list[str]) -> dict:
    base = next((m for m in model_source.MODELS.values() if m.pinned == clf.selection["base_model"]), None)
    if base is None:
        return {"ok": False, "reason": f"базовой модели {clf.selection['base_model']} нет в реестре"}
    model_source.verify_registered(base.directory)
    base_tokenizer = AutoTokenizer.from_pretrained(base.directory)
    texts = normalize_texts(texts)
    ours = clf.tokenizer(texts, truncation=False)["input_ids"]
    theirs = base_tokenizer(texts, truncation=False)["input_ids"]
    differ = sum(a != b for a, b in zip(ours, theirs))
    return {"base_model": base.pinned, "texts": len(texts), "texts_with_different_ids": differ, "ok": differ == 0}


def probabilities(clf: ToxicityClassifier, texts: list[str], dtype) -> np.ndarray:
    ds = ToxicCommentsDataset(texts, None, clf.tokenizer, clf.max_length, clf.truncation)
    loader, order = make_length_sorted_dataloader(ds, clf.tokenizer, clf.run_report["spec"]["eval_batch_size"])
    return predict_proba(clf.model, loader, clf.device, order, dtype)[0]


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=data.PROJECT_ROOT / "models" / "rubert-toxic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="заменить существующий отчёт проверки")
    args = parser.parse_args()
    out_path = data.PROJECT_ROOT / "reports" / "eval" / f"{args.model_dir.name}_check.json"
    try:
        ensure_absent([out_path], args.overwrite)
    except OutputExistsError as error:
        raise SystemExit(str(error)) from error

    # пункт 1: ошибка, если не совпало; LoRA сливается только после проверки в условиях прогона
    clf = ToxicityClassifier.load(args.model_dir, args.device, merge=False)
    report = clf.run_report
    if report is None:
        raise SystemExit(f"{args.model_dir}: пробная модель без ссылки на прогон — проверять нечего")
    threshold = clf.threshold
    parts = splits.load_splits()
    weights = weights_file(args.model_dir)
    checks = {"artifact_matches_run_report": {
        "weights": weights, "sha256": report["checkpoint"][f"{weights.replace('.', '_')}_sha256"],
        "epoch": clf.selection["epoch"], "threshold": threshold, "ok": True}}
    checks["tokenizer_matches_base"] = tokenizer_check(
        clf, [t for s in ("validation", "test") for t in parts[s][data.TEXT_COL]])

    run_on_gpu = report["device"].startswith("cuda") and clf.device.type == "cuda"
    checks["run_conditions"] = {} if run_on_gpu else {"skipped": "прогон шёл на GPU в float16; без GPU не воспроизвести"}
    if run_on_gpu:
        for split in ("validation", "test"):
            p = probabilities(clf, parts[split][data.TEXT_COL].tolist(), torch.float16)
            diff = float(np.abs(p - np.asarray(report["predictions"][split])).max())
            checks["run_conditions"][split] = {"max_abs_diff": round(diff, 7), "atol": RUN_ATOL, "ok": diff <= RUN_ATOL}

    checks["merged_lora"] = is_peft_model(clf.model)
    clf.merge()  # как в predict.py и evaluate.py
    checks["inference_path"] = {}
    for split in ("validation", "test"):
        texts, y = parts[split][data.TEXT_COL].tolist(), parts[split][data.LABEL_COL].to_numpy()
        expected = np.asarray(report["predictions"][split])
        p = clf.predict_proba(texts)  # float32, как в predict.py и evaluate.py
        diff = np.abs(p - expected)
        flips = np.flatnonzero((p >= threshold) != (expected >= threshold))
        cm = classification_metrics(y, (p >= threshold).astype(int), p)["confusion_matrix"]
        eval_path = data.PROJECT_ROOT / "reports" / "eval" / f"{args.model_dir.name}_{split}.json"
        eval_cm = json.loads(eval_path.read_text(encoding="utf-8"))["metrics"]["confusion_matrix"] if eval_path.exists() else None
        run_cm = report[split]["at_calibrated_threshold"]["confusion_matrix"]
        near = all(abs(expected[i] - threshold) <= diff[i] for i in flips)
        same_cm = cm == run_cm and (eval_cm is None or eval_cm == cm)
        checks["inference_path"][split] = {
            "max_abs_diff": round(float(diff.max()), 6), "atol": FLOAT32_ATOL,
            "decision_flips": len(flips), "flips_closer_to_threshold_than_diff": near,
            "confusion_matrix": cm, "run_report_confusion_matrix": run_cm, "eval_report_confusion_matrix": eval_cm,
            # без смены решений матрицы обязаны совпасть; смена допустима только у текстов на пороге
            "ok": bool(diff.max() <= FLOAT32_ATOL and near and (same_cm or len(flips) > 0)),
        }

    test = parts["test"]
    expected = np.asarray(report["predictions"]["test"])
    nearest = np.argsort(np.abs(expected - threshold))[:CLI_TEXTS // 2].tolist()
    picked = list(dict.fromkeys(nearest + list(range(CLI_TEXTS))))[:CLI_TEXTS]
    texts = test[data.TEXT_COL].iloc[picked].tolist()
    in_process = clf.predict_proba(texts)
    # вторая копия модели в predict.py: на GPU с 4 ГБ память родительского процесса нужно освободить
    clf.model.to("cpu")
    if clf.device.type == "cuda":
        torch.cuda.empty_cache()
    out = subprocess.run([sys.executable, str(data.PROJECT_ROOT / "predict.py"), "--model-dir", str(args.model_dir),
                          "--device", str(clf.device.type), *texts], capture_output=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if out.returncode:  # тексты в сообщение не попадают
        raise SystemExit(f"predict.py завершился с кодом {out.returncode}:\n{out.stderr.decode('utf-8', 'replace')[-2000:]}")
    rows = [json.loads(line) for line in out.stdout.decode("utf-8").splitlines() if line.startswith("{")]
    cli_probs = np.array([r["toxic_probability"] for r in rows])
    cli_diff = float(np.abs(cli_probs - in_process).max()) if len(rows) == len(texts) else float("inf")
    cli_decisions_equal = [r["toxic"] for r in rows] == (in_process >= threshold).astype(int).tolist()
    checks["cli"] = {"source_rows": test[data.ID_COL].iloc[picked].astype(int).tolist(), "texts": len(texts),
                     "max_abs_diff": round(cli_diff, 6), "atol": CLI_ATOL,
                     "decisions_equal": cli_decisions_equal, "ok": cli_diff <= CLI_ATOL and cli_decisions_equal}

    ok = all(c.get("ok", True) for c in [checks["artifact_matches_run_report"], checks["tokenizer_matches_base"],
                                         checks["cli"], *checks["inference_path"].values()]
             ) and all(c.get("ok", True) for c in checks["run_conditions"].values() if isinstance(c, dict))
    result = {"environment": capture(neural=True), "model_dir": data.project_relative(args.model_dir), "run_report": clf.selection["run"]["report"],
              "device": str(clf.device), "gpu": torch.cuda.get_device_name(0) if clf.device.type == "cuda" else None,
              "checks": checks, "ok": ok}
    write_json(out_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{'OK' if ok else 'РАСХОЖДЕНИЕ'}: {out_path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
