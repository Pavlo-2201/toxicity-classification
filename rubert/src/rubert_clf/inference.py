"""Загрузка обученной модели и предсказания для сырых текстов.

Артефакт (models/rubert-toxic) при загрузке сверяется с отчётом прогона, на который ссылается
его selection.json: SHA-256 весов, эпоха, порог и параметры токенизации должны совпасть. Так
предсказания и порог загруженной модели — это предсказания и порог оценённого прогона.

LoRA-артефакт (models/rubert-toxic-lora) хранит только адаптер: базовая модель берётся из реестра
model_source по selection.json (файлы сверяются с ревизией), адаптер накладывается через PEFT и по
умолчанию сливается с весами (merge_and_unload) — инференс как у обычной модели.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf import data, model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.finetune import predict_proba
from rubert_clf.runtime import get_device

ADAPTER_CONFIG = "adapter_config.json"


def is_peft_model(model) -> bool:
    """Модель с LoRA-адаптером (у слитой модели атрибут peft_config остаётся, поэтому проверяется тип)."""
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def weights_file(model_dir: Path) -> str:
    return "adapter_model.safetensors" if (Path(model_dir) / ADAPTER_CONFIG).exists() else "model.safetensors"


def verify_run(model_dir: Path, selection: dict) -> dict | None:
    """Отчёт прогона артефакта после сверки с артефактом; None для пробной модели без ссылки на прогон."""
    run = selection.get("run")
    if run is None:
        return None
    report_path = data.PROJECT_ROOT / run["report"]
    if not report_path.exists():
        raise FileNotFoundError(f"Отчёт прогона {report_path} не найден: без него нельзя проверить, что это за веса")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = report.get("checkpoint")
    if checkpoint is None:
        raise ValueError(f"{run['report']}: прогон не сохранял веса (нет поля checkpoint)")
    weights = weights_file(model_dir)
    key = f"{weights.replace('.', '_')}_sha256"
    if key not in checkpoint:
        raise ValueError(f"{run['report']}: в checkpoint нет {key} — отчёт описывает веса другого формата")
    spec = report["spec"]
    expected = {f"{weights} SHA-256": checkpoint[key],
                "эпоха": report["selected_epoch"], "порог": report["threshold"],
                "кандидат и seed": (spec["name"], spec["seed"]),
                "max_length": spec["max_length"], "truncation": spec["truncation"]}
    actual = {f"{weights} SHA-256": data.sha256_file(Path(model_dir) / weights),
              "эпоха": selection["epoch"], "порог": selection["threshold"],
              "кандидат и seed": (run["candidate"], run["seed"]),
              "max_length": selection["config"]["max_length"], "truncation": selection["config"].get("truncation", "head")}
    if "base_model" in report:  # в отчётах до реестра моделей поля нет
        expected["базовая модель"], actual["базовая модель"] = report["base_model"], selection["base_model"]
    changed = [f"{k}: в отчёте {expected[k]!r}, у модели {actual[k]!r}" for k in expected if expected[k] != actual[k]]
    if checkpoint["epoch"] != report["selected_epoch"]:
        changed.append(f"эпоха checkpoint {checkpoint['epoch']} ≠ выбранной {report['selected_epoch']}")
    if changed:
        raise ValueError(f"{model_dir} не совпадает с отчётом прогона {run['report']}:\n  " + "\n  ".join(changed))
    return report


def load_adapter(model_dir: Path, selection: dict):
    """Базовая модель из реестра (с проверкой ревизии) + LoRA-адаптер артефакта, без слияния."""
    from peft import PeftModel

    base = next((m for m in model_source.MODELS.values() if m.pinned == selection["base_model"]), None)
    if base is None:
        raise ValueError(f"Базовой модели {selection['base_model']} нет в реестре model_source")
    adapter = json.loads((Path(model_dir) / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    if (adapter["base_model_name_or_path"], adapter.get("revision")) != (base.model_id, base.revision):
        raise ValueError(f"{ADAPTER_CONFIG}: адаптер обучен на {adapter['base_model_name_or_path']}@{adapter.get('revision')}, "
                         f"а selection.json указывает {base.pinned}")
    model_source.verify_registered(base.directory)
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()  # голова классификатора не инициализирована — её даёт адаптер
    try:
        model = AutoModelForSequenceClassification.from_pretrained(base.directory, num_labels=2)
    finally:
        transformers.logging.set_verbosity(verbosity)
    return PeftModel.from_pretrained(model, model_dir)


class ToxicityClassifier:
    def __init__(self, model, tokenizer, selection: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.selection = selection
        self.device = device
        self.threshold = float(selection["threshold"])
        self.max_length = int(selection["config"]["max_length"])
        self.truncation = selection["config"].get("truncation", "head")

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto", merge: bool = True) -> "ToxicityClassifier":
        """merge — для LoRA-артефакта слить адаптер с весами (на предсказания не влияет, быстрее на CPU)."""
        model_dir = Path(model_dir)
        selection_path = model_dir / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(f"{selection_path} не найден. Обучите модель: python train.py")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run_report = verify_run(model_dir, selection)  # до загрузки весов
        if (model_dir / ADAPTER_CONFIG).exists():
            model = load_adapter(model_dir, selection)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        clf = cls(model, tokenizer, selection, get_device(device))
        clf.run_report = run_report
        if merge:
            clf.merge()
        return clf

    def merge(self) -> None:
        """Слить LoRA-адаптер с весами; для обычной модели ничего не делает."""
        if is_peft_model(self.model):
            self.model = self.model.merge_and_unload().eval()

    def predict_proba(self, texts, batch_size: int = 128) -> np.ndarray:
        """Вероятность токсичного класса (softmax), в порядке входных текстов."""
        texts = list(texts)
        if not texts:
            return np.empty(0)
        ds = ToxicCommentsDataset(texts, None, self.tokenizer, self.max_length, self.truncation)
        loader, order = make_length_sorted_dataloader(ds, self.tokenizer, batch_size)
        return predict_proba(self.model, loader, self.device, order)[0]

    def predict(self, texts, batch_size: int = 128) -> np.ndarray:
        return (self.predict_proba(texts, batch_size) >= self.threshold).astype(int)
