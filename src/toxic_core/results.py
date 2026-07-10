"""Общий формат результатов и защита от случайной перезаписи.

Оценка любой модели на разбиении даёт два файла с одинаковой схемой:

- `<имя>_<разбиение>_predictions.csv` — source_row, toxic, score, probability, pred.
  score — оценка, к которой применяется порог (classical: margin LinearSVC, rubert: P(toxic));
  probability — вероятность токсичности (classical: Platt на OOF train, rubert: softmax);
  pred = score ≥ threshold.
- `<имя>_<разбиение>.json` — модель и артефакт, данные (SHA-256 манифеста и отпечаток), порог и
  протокол его выбора, метрики, бутстреп-интервал F1, SHA-256 файла предсказаний.

Время инференса в общий отчёт не входит: повторная оценка того же артефакта даёт те же байты, и
`write_text_guarded` отличает «без изменений» от настоящего расхождения.
"""

from __future__ import annotations

from toxic_core.environment import capture

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from toxic_core import data, splits
from toxic_core.metrics import bootstrap_f1_ci, classification_metrics, mcnemar, paired_bootstrap_f1_diff

FORMAT = "toxic-eval/1"
PREDICTION_COLUMNS = [data.ID_COL, data.LABEL_COL, "score", "probability", "pred"]
EVAL_DIR = data.REPO_ROOT / "reports" / "eval"


class OutputExistsError(FileExistsError):
    """Результат уже есть и отличается от нового; перезапись — только явно."""


def ensure_absent(paths, overwrite: bool = False, hint: str = "") -> None:
    """Проверка до долгой работы (обучение): выходные файлы и непустые каталоги не должны существовать."""
    existing = []
    for path in map(Path, paths):
        if path.is_dir() and any(path.iterdir()) or path.is_file():
            existing.append(data.relative_to(path))
    if existing and not overwrite:
        raise OutputExistsError("Уже существуют: " + ", ".join(existing) + ". Чтобы не перезаписать готовые "
                                "результаты, укажите другой путь" + (f" ({hint})" if hint else "")
                                + " или --overwrite.")


def write_text_guarded(path: Path, text: str, overwrite: bool = False) -> str:
    """Пишет файл, если его нет; совпадающий по байтам оставляет; отличающийся — только с overwrite."""
    path = Path(path)
    payload = text.encode("utf-8")
    existed = path.exists()
    if existed and path.read_bytes() == payload:
        return "без изменений"
    if existed and not overwrite:
        raise OutputExistsError(f"{data.relative_to(path)} уже существует и отличается от нового результата. "
                                "Укажите другой каталог или --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return "перезаписан" if existed else "записан"


def json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def csv_text(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def predictions_frame(part: pd.DataFrame, scores, probabilities, pred) -> pd.DataFrame:
    probs = np.full(len(part), np.nan) if probabilities is None else np.round(np.asarray(probabilities, dtype=float), 6)
    return pd.DataFrame({data.ID_COL: part[data.ID_COL].to_numpy(), data.LABEL_COL: part[data.LABEL_COL].to_numpy(),
                         "score": np.round(np.asarray(scores, dtype=float), 6), "probability": probs,
                         "pred": np.asarray(pred, dtype=int)})


def evaluate(predictor, split: str, part: pd.DataFrame, limit: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Оценка загруженной модели (адаптер toxic_core.registry) на разбиении в общем формате.

    Порог — сохранённый в артефакте; здесь он только применяется. Решение модели обязано совпасть с
    score ≥ threshold — иначе общий формат описывал бы не то, что модель делает на самом деле.
    """
    texts = part[data.TEXT_COL].tolist()
    scores, probabilities, pred = predictor.predict(texts)
    scores, pred = np.asarray(scores, dtype=float), np.asarray(pred, dtype=int)
    if not np.array_equal(pred, (scores >= predictor.threshold).astype(int)):
        raise RuntimeError(f"{predictor.name}: решения модели не совпадают с score ≥ threshold")
    y = part[data.LABEL_COL].to_numpy()
    info = predictor.describe()
    report = {
        "format": FORMAT,
        "environment": capture(neural=predictor.name == "rubert"),
        "model": predictor.name,
        "split": split,
        "n": int(len(part)),
        "limit": limit,
        "data": {"split_manifest_sha256": splits.MANIFEST_SHA256, "dataset_fingerprint": data.EXPECTED_FINGERPRINT},
        "artifact": info["artifact"],
        "threshold": float(predictor.threshold),
        "score": info["score"],
        "probability": info["probability"],
        "protocol": info["protocol"],
        "metrics": classification_metrics(y, pred, scores),
        "f1_bootstrap_ci": bootstrap_f1_ci(y, pred),
    }
    return report, predictions_frame(part, scores, probabilities, pred)


def eval_paths(out_dir: Path, name: str, split: str) -> tuple[Path, Path]:
    return Path(out_dir) / f"{name}_{split}.json", Path(out_dir) / f"{name}_{split}_predictions.csv"


def save_evaluation(report: dict, predictions: pd.DataFrame, out_dir: Path, name: str, overwrite: bool = False) -> dict:
    """Оба файла оценки; отчёт ссылается на предсказания по SHA-256. Перед записью проверяются оба файла."""
    report_path, pred_path = eval_paths(out_dir, name, report["split"])
    pred_text = csv_text(predictions)
    report = {**report, "predictions_file": pred_path.name,
              "predictions_sha256": hashlib.sha256(pred_text.encode("utf-8")).hexdigest()}
    report_text = json_text(report)
    for path, text in ((report_path, report_text), (pred_path, pred_text)):
        if path.exists() and path.read_bytes() != text.encode("utf-8") and not overwrite:
            raise OutputExistsError(f"{data.relative_to(path)} уже существует и отличается от нового результата. "
                                    "Укажите другой каталог (--out-dir), другое имя (--name) или --overwrite.")
    return {data.relative_to(pred_path): write_text_guarded(pred_path, pred_text, overwrite),
            data.relative_to(report_path): write_text_guarded(report_path, report_text, overwrite)}


def load_evaluation(out_dir: Path, name: str, split: str) -> tuple[dict, pd.DataFrame]:
    """Отчёт и предсказания с проверкой, что файл предсказаний — тот, на который ссылается отчёт."""
    report_path, pred_path = eval_paths(out_dir, name, split)
    for path in (report_path, pred_path):
        if not path.exists():
            raise FileNotFoundError(f"{data.relative_to(path)} не найден. Сначала: toxic evaluate {name} --split {split}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != FORMAT:
        raise ValueError(f"{data.relative_to(report_path)}: формат {report.get('format')!r}, ожидался {FORMAT!r}")
    if data.sha256_file(pred_path) != report["predictions_sha256"]:
        raise ValueError(f"{data.relative_to(pred_path)} не совпадает с SHA-256 из {report_path.name}")
    return report, pd.read_csv(pred_path)


def compare(a: tuple[dict, pd.DataFrame], b: tuple[dict, pd.DataFrame], names: tuple[str, str]) -> dict:
    """Парное сравнение двух оценок в общем формате: одни и те же тексты, метки и манифест обязательны."""
    (report_a, pred_a), (report_b, pred_b) = a, b
    problems = []
    if report_a["split"] != report_b["split"]:
        problems.append(f"разные разбиения: {report_a['split']} и {report_b['split']}")
    if report_a["data"] != report_b["data"]:
        problems.append(f"разные данные: {report_a['data']} и {report_b['data']}")
    pred_a, pred_b = pred_a.sort_values(data.ID_COL).reset_index(drop=True), pred_b.sort_values(data.ID_COL).reset_index(drop=True)
    if not pred_a[data.ID_COL].equals(pred_b[data.ID_COL]):
        problems.append("разные наборы source_row")
    elif not pred_a[data.LABEL_COL].equals(pred_b[data.LABEL_COL]):
        problems.append("разные метки при одинаковых source_row")
    if problems:
        raise ValueError("Оценки несопоставимы: " + "; ".join(problems))
    y, first, second = pred_a[data.LABEL_COL].to_numpy(), pred_a["pred"].to_numpy(), pred_b["pred"].to_numpy()
    summary = lambda r: {"environment": r.get("environment"), "artifact": r["artifact"], "threshold": r["threshold"], "protocol": r["protocol"],
                         "metrics": r["metrics"], "f1_bootstrap_ci": r["f1_bootstrap_ci"]}
    return {
        "format": FORMAT,
        "environment": capture(),
        "split": report_a["split"],
        "n": int(len(y)),
        "data": report_a["data"],
        "models": {names[0]: summary(report_a), names[1]: summary(report_b)},
        "f1_toxic_difference": {"definition": f"F1 {names[1]} − F1 {names[0]}, парный бутстреп (1000, seed 42)",
                                "point": round(report_b["metrics"]["f1_toxic"] - report_a["metrics"]["f1_toxic"], 4),
                                **paired_bootstrap_f1_diff(y, second, first)},
        "mcnemar": {"first": names[0], "second": names[1], **mcnemar(y, first, second)},
        "decisions": {"agree": int((first == second).sum()), "both_wrong": int(((first != y) & (second != y)).sum())},
    }
