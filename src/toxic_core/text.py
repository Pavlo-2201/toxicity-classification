"""Минимальная нормализация текста — общая для обеих моделей."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, схлопывание пробельных символов (в т. ч. хвостового \\n) и обрезка краёв."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalize_texts(texts) -> list[str]:
    return [normalize_text(t) for t in texts]
