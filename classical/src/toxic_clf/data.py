"""Пути классической части.

Источник данных, загрузка и проверки общие для обеих моделей (toxic_core.data) и здесь только
переэкспортируются: модули части обращаются к данным и к своим каталогам через один импорт.
"""

from __future__ import annotations

from pathlib import Path

from toxic_core.data import *  # noqa: F401,F403 — общий источник данных

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # classical/: модели, отчёты и данные проверки этой части
