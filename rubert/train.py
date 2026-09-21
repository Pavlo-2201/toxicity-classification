"""Обучение итоговой модели.

По умолчанию — ruBert-base по протоколу (configs/final_base.json): прогон кандидата протокола
ровно как в экспериментах, сохраняется checkpoint эпохи с лучшим AP на validation, порог — по
калибровочной части. configs/final_lora.json — так же для LoRA-кандидата: сохраняется только
адаптер (models/rubert-toxic-lora), базовая модель берётся из реестра. --smoke и --config с
FinetuneConfig — короткое обучение rubert-tiny2 для smoke-проверок и пользовательских конфигураций.

Готовые артефакт и отчёт прогона не перезаписываются без --overwrite: повторный прогон на GPU
не воспроизводится побитно, и замена опубликованных весов и отчёта должна быть осознанной.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from rubert_clf import data, model_source
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.finetune import FinetuneConfig, build_model, fit, write_json
from rubert_clf.inference import is_peft_model
from rubert_clf.runtime import get_device, set_seed
from toxic_core import splits
from toxic_core.results import OutputExistsError, ensure_absent
from toxic_core.stdio import use_utf8_stdio

DEFAULT_CONFIG = data.PROJECT_ROOT / "configs" / "finetune_best.json"
FINAL_CONFIG = data.PROJECT_ROOT / "configs" / "final_base.json"
SMOKE_CONFIG = data.PROJECT_ROOT / "configs" / "smoke.json"


def export_final(result: dict, candidate: dict, seed: int, protocol_path: Path, output_dir: Path, run_report: Path) -> dict:
    """Артефакт: веса и токенизатор выбранной эпохи, отчёт прогона с SHA-256 весов, selection.json.

    У LoRA сохраняется только адаптер (вместе с обученной головой классификатора). ToxicityClassifier.load
    сверяет артефакт с этим отчётом (src/rubert_clf/inference.py).
    """
    model, tokenizer = result.pop("model"), result.pop("tokenizer")
    adapter = is_peft_model(model)
    if adapter:  # в adapter_config.json — опубликованная модель и ревизия, а не локальный путь
        base = model_source.registered(data.PROJECT_ROOT / candidate["model_dir"])
        model.peft_config["default"].base_model_name_or_path = base.model_id
        model.peft_config["default"].revision = base.revision
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    if adapter:
        # без сетевой проверки словаря базовой модели (эмбеддинги не обучались); шаблонная карточка
        # PEFT с локальным путём к базовой модели не нужна
        model.save_pretrained(output_dir, save_embedding_layers=False)
        (output_dir / "README.md").unlink(missing_ok=True)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    weights = output_dir / ("adapter_model.safetensors" if adapter else "model.safetensors")
    result["checkpoint"] = {"model_dir": data.project_relative(output_dir), "epoch": result["selected_epoch"],
                            f"{weights.name.replace('.', '_')}_sha256": data.sha256_file(weights)}
    write_json(run_report, result)

    selection = {"epoch": result["selected_epoch"], "threshold": result["threshold"],
                 "threshold_rule": result["threshold_rule"], "selection_rule": result["selection_rule"],
                 "config": {k: candidate[k] for k in ("max_length", "truncation", "lr", "batch_size", "epochs", "lora")
                            if k in candidate} | {"seed": seed},
                 "run": {"candidate": candidate["name"], "seed": seed, "report": data.project_relative(run_report),
                         "protocol_sha256": data.sha256_file(protocol_path)},
                 "epochs_validation_ap": {e["epoch"]: e["validation"]["average_precision"] for e in result["epochs"]},
                 "base_model": result["base_model"], "protocol": "AP на validation, порог на калибровочной части train (docs/experiments.md)",
                 "validation_at_0.5": result["validation"]["at_0.5"],
                 "validation_at_threshold": result["validation"]["at_calibrated_threshold"],
                 "test_at_threshold": result["test"]["at_calibrated_threshold"]}
    write_json(output_dir / "selection.json", selection)
    return selection


def train_final(raw: dict, config_path: Path, args) -> None:
    """Итоговая модель по протоколу — выбранный checkpoint экспериментального прогона.

    Кандидат и seed берутся из протокола без изменений (те же эпохи и расписание lr), эпоха — по AP
    на validation, сохраняются веса именно этой эпохи. Отчёт финального прогона хранится отдельно
    в reports/final_runs: checkpoint, порог и метрики относятся к одному обучению.
    """
    from rubert_clf.protocol import RunSpec, calibration_split, long_mask, run

    protocol_path = data.PROJECT_ROOT / raw["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate = next((c for c in protocol["candidates"] if c["name"] == raw["candidate"]), None)
    if candidate is None or raw["seed"] not in protocol["seeds"]:
        raise SystemExit(f"В {protocol_path.name} нет кандидата {raw['candidate']!r} или seed {raw['seed']}")
    output_dir = args.output_dir or data.PROJECT_ROOT / raw.get("output_dir", "models/rubert-toxic")
    run_report = args.run_report or data.PROJECT_ROOT / "reports" / "final_runs" / f"{candidate['name']}__s{raw['seed']}.json"
    guard_outputs([output_dir, run_report], args.overwrite, "--output-dir и --run-report")
    device = get_device(args.device)
    parts = splits.load_splits()
    fit_df, calib_df = calibration_split(parts["train"])
    sets = {"fit": fit_df, "calibration": calib_df, "validation": parts["validation"], "test": parts["test"]}
    masks = {k: long_mask(sets[k][data.TEXT_COL]) for k in ("validation", "test")}
    # CPU-инференс измеряется у первого seed — как в scripts/experiments.py
    spec = RunSpec(**{**candidate, "model_dir": str(data.PROJECT_ROOT / candidate["model_dir"]), "seed": raw["seed"],
                      "extra": {"measure_cpu": raw["seed"] == protocol["seeds"][0], "keep_model": True}})
    print(f"{candidate['name']} seed {raw['seed']}: fit {len(fit_df)}, калибровка {len(calib_df)}, устройство {device}")
    start = time.perf_counter()
    result = run(spec, sets, device, masks)
    result["wall_seconds"] = round(time.perf_counter() - start, 1)
    selection = export_final(result, candidate, raw["seed"], protocol_path, output_dir, run_report)
    write_json(output_dir / "train_metadata.json", {
        "mode": "final", "config_file": config_path.name, "base_model": selection["base_model"],
        "run_report": data.project_relative(run_report),
        "data_manifest_sha256": splits.MANIFEST_SHA256, "data_fingerprint": data.EXPECTED_FINGERPRINT,
        "n_fit": len(fit_df), "n_calibration": len(calib_df), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "total_seconds": result["wall_seconds"], "epochs": result["epochs"],
    })
    print(f"эпоха {result['selected_epoch']} (AP по эпохам {selection['epochs_validation_ap']}), порог {result['threshold']:.4f}; "
          f"validation F1 {result['validation']['at_calibrated_threshold']['f1_toxic']}, "
          f"AP {result['validation']['at_0.5']['average_precision']}")
    print(f"Модель сохранена в {output_dir}; отчёт прогона — {run_report}")


def guard_outputs(paths, overwrite: bool, hint: str) -> None:
    try:
        ensure_absent(paths, overwrite, hint)
    except OutputExistsError as error:
        raise SystemExit(str(error)) from error


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help=f"конфигурация (по умолчанию {FINAL_CONFIG.name})")
    parser.add_argument("--smoke", action="store_true", help=f"малая выборка и 1 эпоха ({SMOKE_CONFIG.name})")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-report", type=Path, default=None,
                        help="отчёт прогона итоговой модели; по умолчанию reports/final_runs/<кандидат>__s<seed>.json")
    parser.add_argument("--base-model-dir", type=Path, default=model_source.MODEL_DIR,
                        help="предобученный энкодер; файлы моделей из реестра сверяются с ревизией")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="по умолчанию checkpoints/<имя папки модели>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="заменить существующие артефакт и отчёт прогона")
    args = parser.parse_args()

    config_path = args.config or (SMOKE_CONFIG if args.smoke else FINAL_CONFIG)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "candidate" in raw:
        train_final(raw, config_path, args)
        return
    config = FinetuneConfig.from_dict(raw)
    output_dir = args.output_dir or data.PROJECT_ROOT / "models" / ("smoke" if args.smoke else "rubert-toxic")
    checkpoint_dir = args.checkpoint_dir or data.PROJECT_ROOT / "checkpoints" / output_dir.name
    guard_outputs([output_dir], args.overwrite, "--output-dir")

    base_dir = args.base_model_dir
    try:  # модель из реестра сверяется с ревизией; своя папка не проверяется
        pinned = model_source.verify_registered(base_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    set_seed(config.seed)
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    parts = splits.load_splits()
    train_df, val_df = parts["train"], parts["validation"]
    if "sample_train" in raw:
        train_df = train_df.sample(raw["sample_train"], random_state=config.seed)
        val_df = val_df.sample(raw["sample_validation"], random_state=config.seed)

    train_ds = ToxicCommentsDataset(train_df[data.TEXT_COL], train_df[data.LABEL_COL], tokenizer, config.max_length)
    val_ds = ToxicCommentsDataset(val_df[data.TEXT_COL], val_df[data.LABEL_COL], tokenizer, config.max_length)
    print(f"train {len(train_ds)}, validation {len(val_ds)}, устройство {device}, конфигурация {config_path.name}")
    model = build_model(base_dir)
    start = time.perf_counter()
    history = fit(model, train_ds, val_ds, tokenizer, config, device, checkpoint_dir=checkpoint_dir, resume=args.resume)
    total = time.perf_counter() - start

    best_dir = checkpoint_dir / "best"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_dir, output_dir)
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    write_json(output_dir / "train_metadata.json", {
        "mode": "smoke" if args.smoke else "full",
        "config_file": config_path.name,
        "base_model": pinned.pinned if pinned else str(base_dir),
        "data_manifest_sha256": splits.MANIFEST_SHA256,
        "data_fingerprint": data.EXPECTED_FINGERPRINT,
        "n_train": len(train_ds), "n_validation": len(val_ds),
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "total_seconds": round(total, 1),
        "history": [{k: v for k, v in h.items() if k != "val_probs"} for h in history],
    })
    print(f"Выбрана эпоха {selection['epoch']}, порог {selection['threshold']:.4f}, "
          f"validation F1 {selection['validation_tuned']['f1_toxic']}")
    print(f"Модель сохранена в {output_dir}")


if __name__ == "__main__":
    main()
