"""Предобученные модели протокола: опубликованные модели Hugging Face с зафиксированными ревизиями.

Для каждой модели записаны ревизия и идентификаторы нужных файлов на Hugging Face Hub: для
LFS-файлов — SHA-256 содержимого, для обычных — git blob SHA-1. Файлы проверяются не только при
скачивании (scripts/download_model.py), но и при каждой загрузке из зарегистрированной папки
(`verify_registered`): обучение и оценка не начнутся на файлах другой ревизии.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rubert_clf import data

MODELS_ROOT = data.PROJECT_ROOT / "models"


@dataclass(frozen=True)
class PinnedModel:
    key: str
    model_id: str
    revision: str
    license: str
    role: str
    files: dict[str, tuple[str, str]]

    @property
    def directory(self) -> Path:
        return MODELS_ROOT / self.key

    @property
    def pinned(self) -> str:
        return f"{self.model_id}@{self.revision}"


MODELS = {m.key: m for m in [
    PinnedModel(
        key="rubert-tiny2", model_id="cointegrated/rubert-tiny2", revision="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        license="MIT", role="энкодер для smoke-проверок и кандидатов tiny_*, эталонный токенизатор длинных текстов",
        files={
            "config.json": ("git-sha1", "26aa82492df46244bd53d5a520aa1d364d8d7966"),
            "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
            "tokenizer.json": ("git-sha1", "9bb4171eb4d2379a4a3ec41c21841823ee229ecc"),
            "tokenizer_config.json": ("git-sha1", "0ca66644d7807a33e4f9ac11d47b7c39e6ce15e0"),
            "vocab.txt": ("git-sha1", "211012a925fc51581997b6d3ef047bcda9a842f4"),
            "model.safetensors": ("sha256", "26ebb6db2a68593c54c74902d7a74f332da66297693f965cc9f1b0af4abf3894"),
        }),
    # в этой ревизии нет tokenizer_config.json и safetensors — только эти три файла
    PinnedModel(
        key="ruBert-base", model_id="ai-forever/ruBert-base", revision="05f37a2ca9e333fd18f30cd0c96c68d274793c69",
        license="Apache-2.0", role="энкодер: кандидаты base_L128 и base_L128_lora, итоговая модель",
        files={
            "config.json": ("git-sha1", "e345c56d236029c22193f7c9ca7d73c28ba9b11c"),
            "vocab.txt": ("git-sha1", "77c387fcb3534a739615ef9832b4ca1b054193ab"),
            "pytorch_model.bin": ("sha256", "6ab6521e029933cfd0731798544bf7d51876905752e760919c94e5eb4e37f851"),
        }),
    PinnedModel(
        key="ready/rubert-tiny-toxicity", model_id="cointegrated/rubert-tiny-toxicity",
        revision="5d37eff844868e243467e4c38898bad46c271af2", license="MIT", role="готовый классификатор (сравнение)",
        files={
            "config.json": ("git-sha1", "71001c4437a3487df5203af263085179c6c0d96e"),
            "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
            "tokenizer.json": ("git-sha1", "713d7dd4ee1d190e0c49fa54bbd83f29b787af6e"),
            "tokenizer_config.json": ("git-sha1", "2a055a9f2e732c6caefa94de8b673723738922e0"),
            "vocab.txt": ("git-sha1", "a1e41485f5de5933a160ff2f18c2254689008be6"),
            "model.safetensors": ("sha256", "e29f120dfe1f0e3bd60823967efe31eb9104ca546e40222fc4a63eaa554ea081"),
        }),
    PinnedModel(
        key="ready/russian_toxicity_classifier", model_id="s-nlp/russian_toxicity_classifier",
        revision="0694e1f99efc08e73479e5c6f06c7bbe393aca89", license="OpenRAIL++", role="готовый классификатор (сравнение)",
        files={
            "config.json": ("git-sha1", "67cafc7c4a515ade7fe9be27183696e4a6117731"),
            "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
            "tokenizer_config.json": ("git-sha1", "b83da8bfc91b3c8e0c9a9253f2dfc38338747c75"),
            "vocab.txt": ("git-sha1", "a95740b522bdec894c9038f8dfb4e405c9e7910d"),
            "model.safetensors": ("sha256", "31321fab9426a686fe01c48c2b3329f2269f2334beecc99664b7518d2c95e9c0"),
        }),
]}

# rubert-tiny2 — модель по умолчанию (smoke, тесты, CI)
DEFAULT = MODELS["rubert-tiny2"]
MODEL_ID, REVISION, LICENSE, FILES = DEFAULT.model_id, DEFAULT.revision, DEFAULT.license, DEFAULT.files
MODEL_CARD_URL = f"https://huggingface.co/{MODEL_ID}"
MODEL_DIR = DEFAULT.directory


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def verify_files(model_dir: Path, files: dict[str, tuple[str, str]]) -> dict[str, bool]:
    """Сверяет файлы папки с ожидаемыми идентификаторами. Возвращает {файл: совпал}."""
    result = {}
    for name, (kind, expected) in files.items():
        path = Path(model_dir) / name
        if not path.exists():
            result[name] = False
            continue
        actual = data.sha256_file(path) if kind == "sha256" else git_blob_sha1(path)
        result[name] = actual == expected
    return result


def verify_model_dir(model_dir: Path = MODEL_DIR) -> dict[str, bool]:
    """Проверка rubert-tiny2 (прежний интерфейс)."""
    return verify_files(model_dir, FILES)


def registered(model_dir) -> PinnedModel | None:
    """Модель реестра, которой соответствует папка, или None для незарегистрированной папки."""
    resolved = Path(model_dir).resolve()
    return next((m for m in MODELS.values() if m.directory.resolve() == resolved), None)


def verify_registered(model_dir) -> PinnedModel | None:
    """Перед загрузкой: файлы зарегистрированной папки должны совпасть с ревизией, иначе ошибка.

    Незарегистрированная папка (например, своя модель или временная в тестах) не проверяется — None.
    """
    model = registered(model_dir)
    if model is None:
        return None
    if not (model.directory / "config.json").exists():
        raise FileNotFoundError(f"{model.directory} не найден. Запустите: python scripts/download_model.py --model {model.key}")
    bad = [name for name, ok in verify_files(model.directory, model.files).items() if not ok]
    if bad:
        raise ValueError(f"Файлы {model.directory} не совпадают с {model.pinned}: {bad}. "
                         f"Скачайте заново: python scripts/download_model.py --model {model.key}")
    return model


def require_model_dir(model_dir: Path = MODEL_DIR) -> Path:
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} не найден. Запустите: python scripts/download_model.py")
    verify_registered(model_dir)
    return model_dir
