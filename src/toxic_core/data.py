"""Источник данных, загрузка и проверки датасета — один для обеих моделей.

Файл s-nlp/rudetoxifier на закреплённом коммите (SHA-256), из него — блок Kaggle-датасета
Russian Language Toxic Comments. Содержимое сверяется по отпечатку, который не зависит от формата CSV.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

__all__ = [
    "REPO_ROOT", "DATA_DIR", "RAW_DIR", "SOURCE_REPO", "SOURCE_COMMIT", "SOURCE_PATH", "SOURCE_URL", "SOURCE_SHA256",
    "SOURCE_FILE", "ORIGINAL_DATASET", "SUBSET_START_ROW", "EXPECTED_ROWS", "EXPECTED_LABEL_COUNTS",
    "EXPECTED_FINGERPRINT", "DATASET_FILE", "TEXT_COL", "LABEL_COL", "ID_COL", "relative_to", "sha256_file",
    "text_sha256", "dataset_fingerprint", "read_source_csv", "extract_subset", "load_dataset", "check_dataset",
    "download", "prepare_dataset",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

SOURCE_REPO = "https://github.com/s-nlp/rudetoxifier"
SOURCE_COMMIT = "aef59899fb99c8da77d4bd6dc69c9c88607e7b6e"
SOURCE_PATH = "data/train/ru_toxic_dataset.csv"
SOURCE_URL = f"https://raw.githubusercontent.com/s-nlp/rudetoxifier/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = "c2c929314de21b69cfa1b809560fa160361c3c202618158134bbcb79b52de0de"
SOURCE_FILE = RAW_DIR / "ru_toxic_dataset.csv"

ORIGINAL_DATASET = {
    "name": "Russian Language Toxic Comments",
    "url": "https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments",
    "version": 1,
    "license": "GPL-2.0",
    "sources": ["2ch.hk", "pikabu.ru"],
}

# Блок Kaggle-датасета внутри объединённого файла: строки [SUBSET_START_ROW, конец).
SUBSET_START_ROW = 148_775
EXPECTED_ROWS = 14_412
EXPECTED_LABEL_COUNTS = {0: 9_586, 1: 4_826}
EXPECTED_FINGERPRINT = "5cce7860d064da67a4cf680b5b33fb0e7b2b92aed04029dda99f56fa68289b0f"

DATASET_FILE = RAW_DIR / "russian_language_toxic_comments.csv"

TEXT_COL = "comment"
LABEL_COL = "toxic"
ID_COL = "source_row"


def relative_to(path, root: Path = REPO_ROOT) -> str:
    """Путь от root в POSIX-виде — для отчётов в репозитории; путь вне root — как есть."""
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Идентификатор текста: SHA-256 от исходной строки в UTF-8 (без нормализации)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Отпечаток содержимого, не зависящий от формата CSV и версии pandas.

    SHA-256 от строк вида «source_row<TAB>sha256(text)<TAB>label\\n» в порядке source_row.
    """
    digest = hashlib.sha256()
    for row, text, label in df.sort_values(ID_COL)[[ID_COL, TEXT_COL, LABEL_COL]].itertuples(index=False):
        digest.update(f"{row}\t{text_sha256(text)}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def read_source_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: тексты вроде «NA» или «null» не должны превращаться в пропуски.
    return pd.read_csv(path, dtype={TEXT_COL: str}, keep_default_na=False)


def extract_subset(source: pd.DataFrame) -> pd.DataFrame:
    """Выделяет блок Kaggle-датасета и приводит метку к int."""
    subset = source.iloc[SUBSET_START_ROW:].copy()
    subset.insert(0, ID_COL, range(SUBSET_START_ROW, SUBSET_START_ROW + len(subset)))
    labels = pd.to_numeric(subset[LABEL_COL], errors="raise")
    if not labels.isin([0.0, 1.0]).all():
        raise ValueError(f"Неожиданные значения меток: {sorted(labels.unique())}")
    subset[LABEL_COL] = labels.astype(int)
    return subset[[ID_COL, TEXT_COL, LABEL_COL]].reset_index(drop=True)


def load_dataset(path: Path = DATASET_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден. Подготовьте данные: toxic prepare")
    df = pd.read_csv(path, dtype={TEXT_COL: str}, keep_default_na=False)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def check_dataset(df: pd.DataFrame) -> dict:
    """Проверки меток, пустых текстов и баланса классов. Возвращает отчёт-словарь."""
    texts = df[TEXT_COL]
    stripped = texts.str.strip()
    counts = df[LABEL_COL].value_counts().sort_index()
    report = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "label_values": sorted(int(v) for v in df[LABEL_COL].unique()),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "toxic_share": round(float(df[LABEL_COL].mean()), 4),
        "imbalance_ratio_neg_to_pos": round(float(counts.get(0, 0) / max(counts.get(1, 0), 1)), 3),
        "missing_texts": int(texts.isna().sum()),
        "empty_texts": int((texts == "").sum()),
        "whitespace_only_texts": int(((stripped == "") & (texts != "")).sum()),
        "texts_with_surrounding_whitespace": int((texts != stripped).sum()),
        "texts_with_internal_newline": int(stripped.str.contains("\n").sum()),
        "texts_without_cyrillic": int((~texts.str.contains("[А-Яа-яЁё]", regex=True)).sum()),
        "texts_with_uppercase": int(texts.str.contains("[A-ZА-ЯЁ]", regex=True).sum()),
        "duplicate_source_rows": int(df[ID_COL].duplicated().sum()),
    }
    report["passed"] = bool(
        report["label_values"] == [0, 1]
        and report["missing_texts"] == 0
        and report["empty_texts"] == 0
        and report["whitespace_only_texts"] == 0
        and report["duplicate_source_rows"] == 0
    )
    return report


def download(url: str, dest: Path, force: bool = False, log=print) -> None:
    if dest.exists() and not force:
        log(f"Уже скачан: {relative_to(dest)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log(f"Скачиваю {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def prepare_dataset(force_download: bool = False, log=print) -> pd.DataFrame:
    """Исходный файл с проверкой SHA-256 → блок датасета с проверкой состава и отпечатка → DATASET_FILE.

    Любое несовпадение — ошибка: датасет на диске всегда тот, на котором построены разбиения.
    """
    download(SOURCE_URL, SOURCE_FILE, force=force_download, log=log)
    actual = sha256_file(SOURCE_FILE)
    if actual != SOURCE_SHA256:
        raise ValueError(f"SHA-256 исходного файла не совпал: ожидался {SOURCE_SHA256}, получен {actual}")
    log(f"SHA-256 исходного файла совпадает: {actual}")

    subset = extract_subset(read_source_csv(SOURCE_FILE))
    counts = subset[LABEL_COL].value_counts().to_dict()
    if len(subset) != EXPECTED_ROWS or counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Неожиданный состав блока: {len(subset)} строк, метки {counts}")
    fingerprint = dataset_fingerprint(subset)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Отпечаток содержимого не совпал: {fingerprint}")
    subset.to_csv(DATASET_FILE, index=False, lineterminator="\n")
    log(f"Отпечаток содержимого совпадает: {fingerprint}")
    log(f"Сохранено {len(subset)} строк в {relative_to(DATASET_FILE)}")
    return subset
