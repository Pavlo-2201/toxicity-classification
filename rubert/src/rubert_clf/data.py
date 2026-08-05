"""Пути части RuBERT.

Источник данных, загрузка и проверки общие для обеих моделей (toxic_core.data) и здесь только
переэкспортируются: модули части обращаются к данным и к своим каталогам через один импорт.
"""

from __future__ import annotations

from pathlib import Path

from toxic_core.data import *  # noqa: F401,F403 — общий источник данных
from toxic_core.data import relative_to

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rubert/: модели, конфигурации и отчёты этой части


def project_relative(path) -> str:
    """Путь от корня части в POSIX-виде — для отчётов в репозитории; путь вне части — как есть."""
    return relative_to(path, PROJECT_ROOT)
