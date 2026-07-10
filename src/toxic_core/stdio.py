"""Кодировка стандартных потоков для всех точек входа."""

from __future__ import annotations

import sys


def use_utf8_stdio() -> None:
    """Переводит stdin/stdout/stderr в UTF-8.

    На Windows Python пишет в pipe и файл в кодировке локали (cp1251): эмодзи и
    редкие символы вызывают UnicodeEncodeError, а UTF-8-читатели видят кракозябры.
    BOM на входе (PowerShell, Блокнот) отбрасывается.
    """
    for stream, encoding in ((sys.stdin, "utf-8-sig"), (sys.stdout, "utf-8"), (sys.stderr, "utf-8")):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=encoding)
