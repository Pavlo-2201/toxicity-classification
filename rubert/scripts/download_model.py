"""Загрузка моделей протокола по зафиксированным ревизиям и проверка хешей файлов.

    python scripts/download_model.py                  # rubert-tiny2 (тесты, CI)
    python scripts/download_model.py --model ruBert-base
    python scripts/download_model.py --all            # все модели протокола (~1.6 ГБ)
    python scripts/download_model.py --verify-only --all

Реестр моделей, ревизии и хеши — src/rubert_clf/model_source.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rubert_clf import model_source as ms
from rubert_clf.cli import use_utf8_stdio


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", choices=sorted(ms.MODELS), help="модель реестра (можно несколько раз)")
    parser.add_argument("--all", action="store_true", help="все модели реестра")
    parser.add_argument("--from-dir", type=Path, help="папка с уже скачанными файлами ревизии (для одной модели)")
    parser.add_argument("--verify-only", action="store_true", help="только проверить уже скачанные файлы")
    args = parser.parse_args()

    keys = sorted(ms.MODELS) if args.all else (args.model or [ms.DEFAULT.key])
    if args.from_dir and len(keys) != 1:
        parser.error("--from-dir — только для одной модели")

    failed = []
    for key in keys:
        model = ms.MODELS[key]
        if not args.verify_only:
            model.directory.mkdir(parents=True, exist_ok=True)
            if args.from_dir:
                for name in model.files:
                    shutil.copyfile(args.from_dir / name, model.directory / name)
            else:
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=model.model_id, revision=model.revision, allow_patterns=list(model.files),
                                  local_dir=model.directory)
        checks = ms.verify_files(model.directory, model.files)
        for name, ok in checks.items():
            print(f"{'OK ' if ok else 'BAD'} {key}/{name}")
        if all(checks.values()):
            print(f"{model.pinned} ({model.license}; {model.role}) -> {model.directory}")
        else:
            failed.append(key)
    if failed:
        raise SystemExit(f"Файлы не совпадают с ревизией: {failed}")


if __name__ == "__main__":
    main()
