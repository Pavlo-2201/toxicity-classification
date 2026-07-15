"""Обработка текста, нужная только классической части; общая нормализация — toxic_core.text."""

from __future__ import annotations

import re

# Сохранённые артефакты (joblib) ссылаются на нормализацию по имени toxic_clf.text.normalize_texts:
# стабильный импорт обеспечивает загрузку сохранённых конвейеров.
from toxic_core.text import normalize_text, normalize_texts  # noqa: F401

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")


def strip_punctuation_texts(texts) -> list[str]:
    """Только для абляции: удаляет пунктуацию и эмоджи, чтобы измерить их вклад."""
    return [_WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", t)).strip() for t in texts]
