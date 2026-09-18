"""Каталог запуска экспериментов: новый запуск отделён от продолжения старого.

Каждый запуск `scripts/experiments.py` пишет в свой каталог (по умолчанию
experiments/runs/<UTC-время>): manifest.json, splits.json и runs/<прогон>.json. В манифесте —
план прогонов, отпечаток процедуры и сессии запуска: что обучено в каждой.

- Новый запуск — только в пустой каталог; уже существующие отчёты не подхватываются.
- Продолжение (`--resume КАТАЛОГ`) — явный режим: отпечаток пересчитывается и должен совпасть с
  манифестом (протокол, данные и калибровочное разбиение, ревизии моделей, код пакета);
  пропускаются только прогоны, отчёты которых записаны этим же запуском.
- Отчёт пишется атомарно (временный файл и замена), поэтому прерванная запись не выглядит готовым
  прогоном.

Окружение (версии библиотек, GPU) записывается в каждую сессию и при продолжении сравнивается с
первой: отличие — предупреждение, а не ошибка; на GPU прогоны и так не воспроизводятся побитно.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch

from rubert_clf import data, model_source, splits

MANIFEST = "manifest.json"
RUNS = "runs"
DEFAULT_ROOT = data.PROJECT_ROOT / "experiments" / "runs"
PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = data.PROJECT_ROOT / "scripts" / "experiments.py"


class ResumeError(ValueError):
    """Каталог нельзя продолжить: другая процедура или чужие отчёты."""


def relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(data.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_dir(root: Path = DEFAULT_ROOT) -> Path:
    return root / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- План и отпечаток процедуры -----------------------------------------------------------

def candidate_run_name(candidate: str, seed: int) -> str:
    return f"{candidate}__s{seed}"


def ready_run_name(spec: dict) -> str:
    return f"ready__{spec['name'].split('/')[-1]}"


def plan(protocol: dict, only: list[str] | None = None, skip_ready: bool = False) -> list[str]:
    names = [c["name"] for c in protocol["candidates"]]
    unknown = sorted(set(only or []) - set(names))
    if unknown:
        raise ValueError(f"Нет в протоколе: {unknown}; кандидаты: {names}")
    runs = [candidate_run_name(c, s) for c in names if not only or c in only for s in protocol["seeds"]]
    if not skip_ready:
        runs += [ready_run_name(spec) for spec in protocol["ready_classifiers"]]
    return runs


def code_sha256() -> str:
    """Хеш исходников пакета и скрипта экспериментов: продолжать можно только тот же код."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")) + [EXPERIMENT_SCRIPT]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def array_sha256(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()


def model_dirs(protocol: dict, runs: list[str]) -> list[Path]:
    """Модели, нужные плану: энкодеры запланированных кандидатов, готовые классификаторы и эталонный токенизатор."""
    from rubert_clf.protocol import REFERENCE_TOKENIZER

    dirs = [REFERENCE_TOKENIZER]
    dirs += [data.PROJECT_ROOT / c["model_dir"] for c in protocol["candidates"]
             if any(r.startswith(c["name"] + "__s") for r in runs)]
    dirs += [data.PROJECT_ROOT / spec["dir"] for spec in protocol["ready_classifiers"] if ready_run_name(spec) in runs]
    return list(dict.fromkeys(dirs))


def fingerprint(protocol_path: Path, protocol: dict, runs: list[str], parts: dict, masks: dict) -> dict:
    """Всё, что должно совпасть у продолжения с исходным запуском.

    Данные к этому моменту уже проверены `splits.load_splits` по зафиксированным хешам; здесь
    добавляются номера строк калибровочной части и маски длинных текстов — они зависят от версий
    sklearn и токенизатора.
    """
    models = {}
    for model_dir in model_dirs(protocol, runs):
        pinned = model_source.verify_registered(model_dir)  # ошибка, если файлы не совпали с ревизией
        models[relative(model_dir)] = pinned.pinned if pinned else "не из реестра"
    return {
        "protocol": {"path": relative(protocol_path), "sha256": data.sha256_file(protocol_path)},
        "data": {"split_manifest_sha256": splits.MANIFEST_SHA256, "dataset_fingerprint": data.EXPECTED_FINGERPRINT,
                 "rows": {name: len(df) for name, df in parts.items()},
                 "calibration_source_rows_sha256": array_sha256(parts["calibration"][data.ID_COL]),
                 "long_texts_sha256": {name: array_sha256(mask) for name, mask in masks.items()}},
        "models": models,
        "code_sha256": code_sha256(),
    }


def environment(device: torch.device) -> dict:
    import sklearn
    import transformers

    try:
        import peft
        peft_version = peft.__version__
    except ImportError:
        peft_version = None
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft_version, "sklearn": sklearn.__version__, "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "platform": platform.platform()}


def differences(expected, actual, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual), key=str):
            out += differences(expected.get(key), actual.get(key), f"{prefix}{key}.")
        return out
    return [] if expected == actual else [f"{prefix.rstrip('.')}: {expected!r} → {actual!r}"]


# --- Создание, продолжение, учёт прогонов ---------------------------------------------------

def create(run_dir: Path, fp: dict, env: dict, runs: list[str]) -> dict:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} не пуст. Новый запуск — в новый каталог; продолжить старый: --resume {run_dir}")
    (run_dir / RUNS).mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_dir.name, "created": now(), "planned": runs, "fingerprint": fp,
                "sessions": [{"started": now(), "mode": "new", "environment": env, "completed": []}]}
    write_json_atomic(run_dir / MANIFEST, manifest)
    return manifest


def completed_runs(manifest: dict) -> set[str]:
    return {name for session in manifest["sessions"] for name in session["completed"]}


def check_reports(run_dir: Path, manifest: dict) -> list[str]:
    """Отчёты каталога должны быть ровно теми, что записаны в манифест этим запуском."""
    problems = []
    done = completed_runs(manifest)
    on_disk = {p.stem for p in (Path(run_dir) / RUNS).glob("*.json")}
    for name in sorted(on_disk - done):
        problems.append(f"{RUNS}/{name}.json не записан этим запуском (нет в manifest.json)")
    for name in sorted(done - on_disk):
        problems.append(f"{RUNS}/{name}.json отмечен в manifest.json, но файла нет")
    for name in sorted(done & on_disk):
        report = read_json(Path(run_dir) / RUNS / f"{name}.json")
        if report.get("run_id") != manifest["run_id"]:
            problems.append(f"{RUNS}/{name}.json: run_id {report.get('run_id')!r}, ожидался {manifest['run_id']!r}")
        if name not in manifest["planned"]:
            problems.append(f"{RUNS}/{name}.json нет в плане запуска")
    return problems


def resume(run_dir: Path, fp: dict, env: dict) -> dict:
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST).exists():
        raise ResumeError(f"В {run_dir} нет {MANIFEST}: это не каталог запуска experiments.py")
    manifest = read_json(run_dir / MANIFEST)
    changed = differences(manifest["fingerprint"], fp)
    if changed:
        raise ResumeError("Процедура отличается от исходного запуска, продолжать нельзя — начните новый:\n  "
                          + "\n  ".join(changed))
    problems = check_reports(run_dir, manifest)
    if problems:
        raise ResumeError("Отчёты каталога не согласованы с manifest.json:\n  " + "\n  ".join(problems))
    manifest["sessions"].append({"started": now(), "mode": "resume", "environment": env, "completed": []})
    write_json_atomic(run_dir / MANIFEST, manifest)
    return manifest


def environment_changes(manifest: dict) -> list[str]:
    return differences(manifest["sessions"][0]["environment"], manifest["sessions"][-1]["environment"])


def save_report(run_dir: Path, manifest: dict, name: str, report: dict) -> None:
    """Отчёт прогона и отметка в текущей сессии манифеста."""
    report = {"run_id": manifest["run_id"], "session": len(manifest["sessions"]), **report}
    write_json_atomic(Path(run_dir) / RUNS / f"{name}.json", report)
    manifest["sessions"][-1]["completed"].append(name)
    write_json_atomic(Path(run_dir) / MANIFEST, manifest)


# --- Проверка готового запуска ----------------------------------------------------------------

def verify(run_dir: Path, trained_in_last_session: bool = False) -> list[str]:
    """Проблемы запуска (пустой список — всё в порядке).

    Все запланированные прогоны есть, каждый записан ровно в одной сессии, у кандидатов — все эпохи
    протокола с ненулевым временем обучения и конечным loss. С trained_in_last_session все прогоны
    должны быть выполнены в последней сессии: так ручной workflow подтверждает, что обучение было
    в этом задании, а не подхвачено из готовых файлов.
    """
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST).exists():
        return [f"нет {run_dir / MANIFEST}"]
    manifest = read_json(run_dir / MANIFEST)
    problems = check_reports(run_dir, manifest)
    names = [n for s in manifest["sessions"] for n in s["completed"]]
    problems += [f"{n} записан в нескольких сессиях" for n in sorted({n for n in names if names.count(n) > 1})]
    problems += [f"{n}: прогон не выполнен" for n in manifest["planned"] if n not in names]
    if trained_in_last_session:
        last = set(manifest["sessions"][-1]["completed"])
        problems += [f"{n}: выполнен не в последней сессии" for n in manifest["planned"] if n in names and n not in last]
    for name in manifest["planned"]:
        path = run_dir / RUNS / f"{name}.json"
        if name.startswith("ready__") or not path.exists():
            continue
        report = read_json(path)
        epochs = report.get("epochs", [])
        if len(epochs) != report["spec"]["epochs"]:
            problems.append(f"{name}: эпох {len(epochs)} из {report['spec']['epochs']}")
        if not all(e["train_seconds"] > 0 and math.isfinite(e["train_loss"]) for e in epochs):
            problems.append(f"{name}: эпоха без обучения (нулевое время или нечисловой loss)")
    return problems
