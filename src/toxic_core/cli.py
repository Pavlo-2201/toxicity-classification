"""Единая командная строка проекта: `toxic <команда> [classical|rubert] ...` (или `python -m toxic_core`).

  prepare  [classical|rubert]  данные с проверкой хешей, сверка разбиений; для rubert ещё энкодер
  train    classical|rubert    обучение; остальные аргументы передаются <часть>/train.py как есть
  predict  classical|rubert    предсказания для текстов: CSV с колонками score, probability, toxic
  evaluate classical|rubert    оценка на validation/test в общем формате (reports/eval/)
  compare  A B                 парное сравнение двух оценок на одних и тех же текстах
  check                        согласованность данных, разбиений, отчётов и артефактов обеих частей

Существующие результаты не перезаписываются: совпадающий по байтам файл остаётся, отличающийся —
только с --overwrite (у train — тот же флаг в аргументах части).
"""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json
import runpy
import sys
from pathlib import Path

import pandas as pd

from toxic_core import data, registry, results, splits
from toxic_core.stdio import use_utf8_stdio

DATA_REPORTS = data.REPO_ROOT / "reports" / "data"


def run_script(script: Path, args: list[str]) -> None:
    """Запуск скрипта части в этом же процессе, как `python <script> <args>`."""
    argv = sys.argv
    sys.argv = [str(script), *args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = argv


# --- prepare и check --------------------------------------------------------------------------

def data_check_report(df: pd.DataFrame) -> dict:
    return {
        "environment": capture(),
        "dataset": data.ORIGINAL_DATASET,
        "source": {"repo": data.SOURCE_REPO, "commit": data.SOURCE_COMMIT, "path": data.SOURCE_PATH,
                   "sha256": data.SOURCE_SHA256, "subset_start_row": data.SUBSET_START_ROW},
        "fingerprint": data.dataset_fingerprint(df),
        "checks": data.check_dataset(df),
    }


def cmd_prepare(args) -> None:
    df = data.prepare_dataset(force_download=args.force_download)
    if args.rebuild_splits:
        manifest = splits.build_manifest(df)
        status = results.write_text_guarded(splits.MANIFEST_PATH, results.csv_text(manifest), args.overwrite)
        print(f"{data.relative_to(splits.MANIFEST_PATH)}: {status}")
        info = results.json_text(splits.fingerprint_info(df))
        print(f"{data.relative_to(splits.FINGERPRINT_PATH)}: "
              f"{results.write_text_guarded(splits.FINGERPRINT_PATH, info, args.overwrite)}")
    checks = data_check_report(df)
    split_check = {**splits.verify(df), "environment": capture()}
    for name, report in (("data_check.json", checks), ("split_check.json", split_check)):
        path = DATA_REPORTS / name
        print(f"{data.relative_to(path)}: {results.write_text_guarded(path, results.json_text(report), args.overwrite)}")
    if not checks["checks"]["passed"] or not split_check["passed"]:
        raise SystemExit("Проверка данных или разбиений не пройдена: " + json.dumps(
            {"data": checks["checks"]["passed"], "splits": split_check}, ensure_ascii=False))
    counts = {name: value["rows"] for name, value in split_check["split_counts"].items()}
    print(f"Разбиения сверены с манифестом (SHA-256 {splits.MANIFEST_SHA256[:12]}…): {counts}")
    if args.model:
        registry.adapter(args.model).prepare(all_models=args.all_models)


def manifest_references(root: Path) -> dict[str, set[str]]:
    """Все значения ключей *manifest_sha256 в JSON-отчётах каталога: {файл: {sha, ...}}."""
    found: dict[str, set[str]] = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("manifest_sha256") and isinstance(value, str):
                    found.setdefault(path, set()).add(value)
                walk(value, path)
        elif isinstance(node, list):
            for value in node:
                walk(value, path)

    for file in sorted(root.rglob("*.json")):
        if ".venv" in file.parts or "local" in file.parts:
            continue
        walk(json.loads(file.read_text(encoding="utf-8")), data.relative_to(file))
    return found


def consistency_report() -> dict:
    """Обе части читают одни данные и один манифест; все отчёты и артефакты ссылаются на него."""
    from toxic_clf import data as classical_data
    from rubert_clf import data as rubert_data  # без torch: модуль данных не тянет зависимости модели

    report: dict = {"splits": splits.verify()}
    report["shared_paths"] = {
        part: all(getattr(module, name) == getattr(data, name) for name in ("DATASET_FILE", "SOURCE_FILE", "EXPECTED_FINGERPRINT"))
        for part, module in (("classical", classical_data), ("rubert", rubert_data))
    }
    roots = [data.REPO_ROOT / "reports", classical_data.PROJECT_ROOT / "reports", rubert_data.PROJECT_ROOT / "reports",
             classical_data.PROJECT_ROOT / "models", rubert_data.PROJECT_ROOT / "models"]
    references = {}
    for root in roots:
        if root.exists():
            references.update(manifest_references(root))
    report["manifest_references"] = {"files": len(references),
                                     "other_manifest": sorted(f for f, v in references.items() if v != {splits.MANIFEST_SHA256})}
    evaluations = []
    for path in sorted(results.EVAL_DIR.glob("*_predictions.csv")):
        name, split = path.name[: -len("_predictions.csv")].rsplit("_", 1)
        report_eval, _ = results.load_evaluation(results.EVAL_DIR, name, split)  # проверяет SHA-256 предсказаний
        evaluations.append({"name": name, "split": split, "data_ok": report_eval["data"]["split_manifest_sha256"] == splits.MANIFEST_SHA256})
    report["evaluations"] = evaluations
    report["passed"] = bool(report["splits"]["passed"] and all(report["shared_paths"].values())
                            and not report["manifest_references"]["other_manifest"] and all(e["data_ok"] for e in evaluations))
    return report


def cmd_check(args) -> None:
    report = consistency_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Проверка согласованности не пройдена")


# --- train, predict, evaluate, compare --------------------------------------------------------

def cmd_train(args) -> None:
    run_script(registry.adapter(args.model).TRAIN_SCRIPT, args.train_args)


def read_texts(path: str, column: str) -> pd.DataFrame:
    if path == "-":
        lines = sys.stdin.read().splitlines()
    elif Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={column: str}, keep_default_na=False)
        if column not in df.columns:
            raise SystemExit(f"В {path} нет колонки {column!r}")
        return df
    else:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return pd.DataFrame({column: [line for line in lines if line.strip()]})


def cmd_predict(args) -> None:
    if bool(args.text) == bool(args.input):
        raise SystemExit("укажите либо --text, либо --input")
    df = pd.DataFrame({args.column: args.text}) if args.text else read_texts(args.input, args.column)
    if args.output:
        results.ensure_absent([args.output], args.overwrite)
    predictor = registry.adapter(args.model).load(args.artifact, args.device)
    scores, probabilities, pred = predictor.predict(df[args.column].tolist())
    df["score"] = pd.Series(scores, index=df.index).round(4)
    if probabilities is not None:
        df["probability"] = pd.Series(probabilities, index=df.index).round(4)
    df["toxic"] = pred
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, lineterminator="\n")
        print(f"Сохранено {len(df)} предсказаний в {args.output}")
    else:
        df.to_csv(sys.stdout, index=False, lineterminator="\n")


def default_name(args) -> str:
    name = args.model if args.artifact is None else f"{args.model}-{Path(args.artifact).stem}"
    return f"{name}-first{args.limit}" if args.limit else name


def cmd_evaluate(args) -> None:
    name = args.name or default_name(args)
    if "_" in name:
        raise SystemExit("в имени оценки не должно быть «_»: по нему разделяются имя и разбиение")
    predictor = registry.adapter(args.model).load(args.artifact, args.device)
    trained_on = predictor.describe().get("training_manifest_sha256")
    if trained_on != splits.MANIFEST_SHA256:
        raise SystemExit(f"Артефакт обучен на манифесте {trained_on}, а разбиения проекта — {splits.MANIFEST_SHA256}")
    part = splits.load_splits()[args.split]
    if args.limit:
        part = part.head(args.limit)
    report, predictions = results.evaluate(predictor, args.split, part, args.limit)
    written = results.save_evaluation(report, predictions, args.out_dir, name, args.overwrite)
    m = report["metrics"]
    print(f"{name} на {args.split} (n={report['n']}, порог {report['threshold']:.6g}): F1 toxic {m['f1_toxic']} "
          f"[{report['f1_bootstrap_ci']['low']}; {report['f1_bootstrap_ci']['high']}], AP {m.get('average_precision')}")
    for path, status in written.items():
        print(f"{path}: {status}")


def cmd_compare(args) -> None:
    first = results.load_evaluation(args.out_dir, args.first, args.split)
    second = results.load_evaluation(args.out_dir, args.second, args.split)
    report = results.compare(first, second, (args.first, args.second))
    path = Path(args.out_dir) / f"compare-{args.first}-vs-{args.second}_{args.split}.json"
    status = results.write_text_guarded(path, results.json_text(report), args.overwrite)
    diff, test = report["f1_toxic_difference"], report["mcnemar"]
    print(f"F1 {args.second} − {args.first}: {diff['point']:+.4f} [{diff['low']}; {diff['high']}], "
          f"Макнемар p = {test['p_value']} ({test['only_first_correct']} / {test['only_second_correct']})")
    print(f"{data.relative_to(path)}: {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toxic", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    models = sorted(registry.MODELS)

    p = sub.add_parser("prepare", help="данные, проверка хешей и разбиений; для rubert — ещё модель")
    p.add_argument("model", nargs="?", choices=models, help="что подготовить сверх данных (rubert: энкодер rubert-tiny2)")
    p.add_argument("--all-models", action="store_true", help="rubert: все модели протокола (≈2 ГБ)")
    p.add_argument("--force-download", action="store_true", help="скачать исходный файл заново")
    p.add_argument("--rebuild-splits", action="store_true", help="пересобрать манифест разбиений и data_fingerprint.json")
    p.add_argument("--overwrite", action="store_true", help="разрешить замену отличающихся отчётов и манифеста")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("check", help="согласованность данных, разбиений, отчётов и артефактов обеих частей")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("train", help="обучение (аргументы после модели — аргументы <часть>/train.py)",
                       description="Аргументы после имени модели передаются <часть>/train.py без изменений: "
                                   "toxic train classical --help, toxic train rubert --help.")
    p.add_argument("model", choices=models)
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="предсказания для текстов")
    p.add_argument("model", choices=models)
    p.add_argument("--artifact", type=Path, help="артефакт модели (по умолчанию — итоговый артефакт части)")
    p.add_argument("--text", action="append", help="текст (можно несколько раз)")
    p.add_argument("--input", help=".csv с колонкой текста, .txt по тексту на строку или - для stdin")
    p.add_argument("--column", default=data.TEXT_COL)
    p.add_argument("--output", type=Path, help="CSV; по умолчанию stdout")
    p.add_argument("--device", default="auto", help="rubert: auto, cpu или cuda")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="оценка на разбиении в общем формате")
    p.add_argument("model", choices=models)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--artifact", type=Path)
    p.add_argument("--name", help="имя оценки в именах файлов (по умолчанию модель и, если задан, артефакт)")
    p.add_argument("--out-dir", type=Path, default=results.EVAL_DIR)
    p.add_argument("--limit", type=int, help="только первые N текстов разбиения (проверка, не результат)")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="парное сравнение двух оценок")
    p.add_argument("first", help="имя первой оценки (например classical)")
    p.add_argument("second", help="имя второй оценки (например rubert)")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--out-dir", type=Path, default=results.EVAL_DIR)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (results.OutputExistsError, registry.MissingDependencies, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()
