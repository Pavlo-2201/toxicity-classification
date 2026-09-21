"""Модели, доступные через общий интерфейс.

Адаптер модели — модуль её пакета с одинаковым набором имён:

- NAME, PROJECT_ROOT (каталог части), DEFAULT_ARTIFACT, TRAIN_SCRIPT;
- load(artifact, device) -> объект с name, threshold, predict(texts) -> (score, probability, pred)
  и describe() -> {"artifact", "score", "probability", "protocol", "training_manifest_sha256"};
- prepare(all_models, log) — подготовка того, что нужно модели сверх общих данных.

Адаптер импортируется только при обращении к модели: классическая часть работает без PyTorch.
"""

from __future__ import annotations

import importlib

MODELS = {"classical": "toxic_clf.adapter", "rubert": "rubert_clf.adapter"}
RUBERT_PACKAGES = {"torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "peft", "accelerate"}


class MissingDependencies(RuntimeError):
    pass


def adapter(name: str):
    if name not in MODELS:
        raise ValueError(f"Неизвестная модель {name!r}; есть: {', '.join(MODELS)}")
    try:
        return importlib.import_module(MODELS[name])
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] in RUBERT_PACKAGES:
            raise MissingDependencies(
                f"Для модели {name} не установлен {error.name}. Зависимости RuBERT ставятся отдельно: "
                "pip install -r requirements/rubert.txt (или pip install -e \".[rubert]\")") from error
        raise
