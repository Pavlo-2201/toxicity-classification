"""Протокол: выбор эпохи и конфигурации отделён от настройки порога.

- Из train заранее отделяется калибровочная часть (15 %, стратифицированно, seed 42).
  Модель обучается на остатке (fit).
- Эпоха выбирается по average precision на validation: метрика не зависит от порога.
- Порог подбирается только на калибровочной части (максимум F1 токсичного класса).
- Test оценивается и при пороге 0.5, и при подобранном пороге.
- Длинные тексты — тексты test/validation длиннее LONG_TOKENS токенов эталонного
  токенизатора rubert-tiny2: подмножество одно и то же для всех моделей.

После каждой эпохи считаются вероятности validation, калибровочной части и test, а эпоха
выбирается по validation после обучения. Test в выборе не участвует. Веса на диск не пишутся;
с extra["keep_model"] веса лучшей по AP эпохи держатся в памяти, и возвращается модель именно
этой эпохи — так итоговый артефакт совпадает с оценённым прогоном (train.py).

У LoRA-кандидата CPU-инференс измеряется дважды: с адаптером, как обучали, и после слияния адаптера
с весами (merge_and_unload) — так модель используется на практике. Для слитой модели записано и
расхождение вероятностей test с несмёрженной.
"""

from __future__ import annotations

from toxic_core.environment import capture

import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf import data, model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_bucketed_dataloader, make_length_sorted_dataloader
from rubert_clf.finetune import FinetuneConfig, amp_dtype, class_weights, make_optimizer, make_scaler, make_scheduler, \
    predict_proba, train_epoch
from rubert_clf.runtime import set_seed
from toxic_core.metrics import best_f1_threshold, classification_metrics

CALIB_SHARE = 0.15
CALIB_SEED = 42
LONG_TOKENS = 128
REFERENCE_TOKENIZER = data.PROJECT_ROOT / "models" / "rubert-tiny2"


def calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit, calib = train_test_split(train, test_size=CALIB_SHARE, random_state=CALIB_SEED,
                                  stratify=train[data.LABEL_COL])
    return fit.sort_values(data.ID_COL).reset_index(drop=True), calib.sort_values(data.ID_COL).reset_index(drop=True)


def long_mask(texts, tokenizer=None) -> np.ndarray:
    if tokenizer is None:
        model_source.verify_registered(REFERENCE_TOKENIZER)
        tokenizer = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
    from toxic_core.text import normalize_texts

    lengths = [len(ids) for ids in tokenizer(normalize_texts(texts), truncation=False)["input_ids"]]
    return np.asarray(lengths) > LONG_TOKENS


@dataclass
class RunSpec:
    name: str
    model_dir: str
    max_length: int = 128
    truncation: str = "head"
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 128
    class_weight: str = "none"
    lora: dict | None = None
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def finetune_config(self) -> FinetuneConfig:
        return FinetuneConfig(max_length=self.max_length, batch_size=self.batch_size, eval_batch_size=self.eval_batch_size,
                              epochs=self.epochs, lr=self.lr, class_weight=self.class_weight, seed=self.seed)


def build_classifier(spec: RunSpec) -> tuple[nn.Module, int, int]:
    model = AutoModelForSequenceClassification.from_pretrained(spec.model_dir, num_labels=2)
    if spec.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **spec.lora))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def threshold_free(y, probs) -> dict:
    return {"average_precision": round(float(average_precision_score(y, probs)), 4),
            "log_loss": round(float(log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))), 4)}


def evaluate_split(y, probs, threshold: float, long: np.ndarray | None = None) -> dict:
    out = {"at_0.5": classification_metrics(y, (probs >= 0.5).astype(int), probs),
           "at_calibrated_threshold": classification_metrics(y, (probs >= threshold).astype(int), probs)}
    if long is not None and long.any():
        out["long_texts"] = {"n": int(long.sum()), "n_toxic": int(y[long].sum()),
                             "average_precision": round(float(average_precision_score(y[long], probs[long])), 4),
                             "f1_at_calibrated_threshold": classification_metrics(
                                 y[long], (probs[long] >= threshold).astype(int), probs[long])["f1_toxic"]}
    return out


def run(spec: RunSpec, parts: dict[str, pd.DataFrame], device: torch.device, long_masks: dict[str, np.ndarray],
        log=print) -> dict:
    """Обучение одной конфигурации с одним seed и оценка по протоколу."""
    set_seed(spec.seed)
    # файлы зарегистрированной модели сверяются с ревизией до загрузки
    pinned = model_source.verify_registered(spec.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_dir)
    config = spec.finetune_config()
    sets = {name: ToxicCommentsDataset(parts[name][data.TEXT_COL], parts[name][data.LABEL_COL], tokenizer,
                                       spec.max_length, spec.truncation) for name in ("fit", "calibration", "validation", "test")}
    model, trainable, total = build_classifier(spec)
    model.to(device)
    loader = make_bucketed_dataloader(sets["fit"], tokenizer, config.batch_size, seed=config.seed)
    eval_loaders = {name: make_length_sorted_dataloader(sets[name], tokenizer, config.eval_batch_size)
                    for name in ("calibration", "validation", "test")}
    weights = class_weights(sets["fit"].labels, config.class_weight)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    dtype = amp_dtype(config, device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, len(loader))
    scaler = make_scaler(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epochs, best_state = [], None
    keep_model = spec.extra.get("keep_model", False)
    for epoch in range(1, config.epochs + 1):
        loader.batch_sampler.set_epoch(epoch)
        stats = train_epoch(model, loader, optimizer, device, loss_fn=loss_fn, scheduler=scheduler, scaler=scaler,
                            max_grad_norm=config.max_grad_norm, dtype=dtype)
        probs = {}
        started = time.perf_counter()
        for name, (eval_loader, order) in eval_loaders.items():
            probs[name], _ = predict_proba(model, eval_loader, device, order, dtype)
        eval_seconds = time.perf_counter() - started
        y_val = np.asarray(sets["validation"].labels)
        row = {"epoch": epoch, "train_loss": round(stats["train_loss"], 4), "train_seconds": round(stats["seconds"], 2),
               "train_texts_per_second": round(len(sets["fit"]) / stats["seconds"], 1),
               "eval_seconds_3_splits": round(eval_seconds, 2), "validation": threshold_free(y_val, probs["validation"]),
               "probs": probs}
        if keep_model and (not epochs or row["validation"]["average_precision"] > max(
                e["validation"]["average_precision"] for e in epochs)):
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        epochs.append(row)
        log(f"{spec.name} seed {spec.seed} epoch {epoch}: loss {row['train_loss']} val AP {row['validation']['average_precision']} "
            f"logloss {row['validation']['log_loss']} ({row['train_seconds']} c)")

    # при равном AP — более ранняя эпоха; то же правило, что при сохранении best_state выше
    best = max(epochs, key=lambda r: (r["validation"]["average_precision"], -r["epoch"]))
    if keep_model:
        model.load_state_dict(best_state)
        # веса выбранной эпохи восстановлены: вероятности validation должны совпасть с посчитанными на ней
        loader_val, order_val = eval_loaders["validation"]
        restored, _ = predict_proba(model, loader_val, device, order_val, dtype)
        if not np.allclose(restored, best["probs"]["validation"], atol=1e-4):
            raise RuntimeError("Восстановленные веса лучшей эпохи не воспроизводят её вероятности на validation")
    y = {name: np.asarray(sets[name].labels) for name in ("calibration", "validation", "test")}
    threshold, calib_f1 = best_f1_threshold(y["calibration"], best["probs"]["calibration"])
    result = {
        # путь к энкодеру — от корня проекта: отчёт не зависит от машины, где шёл прогон
        "environment": capture(neural=True),
        "spec": asdict(spec) | {"model_dir": data.project_relative(spec.model_dir)},
        "base_model": pinned.pinned if pinned else f"не из реестра: {spec.model_dir}",
        "parameters": {"trainable": int(trainable), "total": int(total)},
        "device": str(device),
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "truncated_share": {name: round(ds.truncated_share, 4) for name, ds in sets.items()},
        "selected_epoch": best["epoch"],
        "selection_rule": "максимум AP на validation (порог не участвует)",
        "threshold": float(threshold),
        "threshold_rule": "максимум F1 токсичного класса на калибровочной части train",
        "calibration_f1_at_threshold": round(float(calib_f1), 4),
        "epochs": [{k: v for k, v in r.items() if k != "probs"} for r in epochs],
        "validation": evaluate_split(y["validation"], best["probs"]["validation"], threshold, long_masks.get("validation")),
        "test": evaluate_split(y["test"], best["probs"]["test"], threshold, long_masks.get("test")),
        "predictions": {name: np.round(best["probs"][name], 5).tolist() for name in ("validation", "test")},
    }
    test_texts = parts["test"][data.TEXT_COL].tolist()
    result["inference"] = measure_inference(model, tokenizer, test_texts, spec, device, dtype)
    if spec.extra.get("measure_cpu") and device.type != "cpu":
        cpu = torch.device("cpu")
        result["inference_cpu"], probs_cpu = measure_inference(model.to(cpu).float(), tokenizer, test_texts, spec, cpu,
                                                               None, return_probs=True)
        if spec.lora:  # копия: возвращаемая модель остаётся с адаптером, его и сохраняет train.py
            result["inference_cpu_merged"] = measure_merged(model, tokenizer, test_texts, spec, probs_cpu, threshold)
    if spec.extra.get("keep_model"):
        result["model"], result["tokenizer"] = model, tokenizer
        return result
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def measure_merged(model, tokenizer, texts: list[str], spec: RunSpec, probs_unmerged: np.ndarray, threshold: float) -> dict:
    """LoRA после слияния с весами на CPU: задержка и совпадение с моделью с адаптером."""
    merged = copy.deepcopy(model).merge_and_unload()
    cpu = torch.device("cpu")
    stats, probs = measure_inference(merged, tokenizer, texts, spec, cpu, None, return_probs=True)
    stats.update({"parameters": int(sum(p.numel() for p in merged.parameters())),
                  "test_max_abs_diff_vs_adapter": round(float(np.abs(probs - probs_unmerged).max()), 7),
                  "test_decision_flips_vs_adapter": int(((probs >= threshold) != (probs_unmerged >= threshold)).sum())})
    del merged
    return stats


@torch.no_grad()
def measure_inference(model, tokenizer, texts: list[str], spec: RunSpec, device, dtype, n_single: int = 200,
                      return_probs: bool = False):
    """Пакетный инференс всего test на устройстве обучения и по одному тексту (n_single текстов)."""
    model.eval()
    ds = ToxicCommentsDataset(texts, None, tokenizer, spec.max_length, spec.truncation)
    loader, order = make_length_sorted_dataloader(ds, tokenizer, spec.eval_batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    probs, _ = predict_proba(model, loader, device, order, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started
    single = []
    for text in texts[:n_single]:
        one = ToxicCommentsDataset([text], None, tokenizer, spec.max_length, spec.truncation)
        ids = torch.tensor([one.input_ids[0]], device=device)
        t = time.perf_counter()
        from rubert_clf.finetune import autocast

        with autocast(device, dtype):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        if device.type == "cuda":
            torch.cuda.synchronize()
        single.append((time.perf_counter() - t) * 1000)
    single.sort()
    stats = {"device": str(device), "batch_ms_per_1000_texts": round(1000 * batch_seconds / len(texts) * 1000, 1),
             "single_text_ms_p50": round(single[len(single) // 2], 2),
             "single_text_ms_p95": round(single[int(len(single) * 0.95) - 1], 2), "single_texts": len(single),
             "note": "одиночный текст: только прямой проход модели, без токенизации"}
    return (stats, probs) if return_probs else stats
