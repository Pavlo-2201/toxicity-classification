"""Без зависимостей RuBERT (установка только классической части) его тесты не собираются.

CI с установленным RuBERT это не скрывает: там шаг проверяет, что torch импортируется.
"""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["rubert/*"]


def pytest_report_header(config):
    return None if not collect_ignore_glob else "rubert/tests пропущены: torch не установлен (requirements/rubert.txt)"
