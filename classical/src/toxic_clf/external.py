"""Внешний набор для оценки: комментарии ok.ru (Toxic Russian Comments, OK ML Cup).

Используется только test-часть зеркала на Hugging Face с закреплённой ревизией и
SHA-256. Тексты в нём приведены к нижнему регистру. Метка 1 — комментарий с
оскорблением, угрозой или непристойностью (классы INSULT, THREAT, OBSCENITY соревнования),
0 — нейтральный. Это не то же определение токсичности, что в 2ch/pikabu: часть
расхождений — разница разметки, а не ошибки модели.
"""

from __future__ import annotations

import re
import shutil
import urllib.request

import pandas as pd

from toxic_clf import data

OKRU = {
    "name": "Toxic Russian Comments (ok.ru)",
    "original": "https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments",
    "mirror_repo": "AlexSham/Toxic_Russian_Comments",
    "revision": "a72ec0ab553d92d14a48bef0c0f4927c950bcf4c",
    "file": "test.jsonl",
    "sha256": "5ea5d3382e19377af885f87f56c614536016ec9f0f70b2394b30e4062e4eb95a",
    "rows": 24_829,
}
OKRU_URL = (f"https://huggingface.co/datasets/{OKRU['mirror_repo']}/resolve/{OKRU['revision']}/{OKRU['file']}")
OKRU_FILE = data.PROJECT_ROOT / "data" / "external" / "okru_test.jsonl"

_NON_WORD = re.compile(r"[^0-9a-zа-я]+")


def overlap_key(text: str) -> str:
    """Ключ для поиска одинаковых текстов: регистр, ё/е, пунктуация и пробелы не различаются."""
    return _NON_WORD.sub(" ", text.lower().replace("ё", "е")).strip()


def download_okru(force: bool = False) -> None:
    if OKRU_FILE.exists() and not force:
        return
    OKRU_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OKRU_FILE.with_suffix(".part")
    with urllib.request.urlopen(OKRU_URL, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(OKRU_FILE)


def load_okru() -> pd.DataFrame:
    actual = data.sha256_file(OKRU_FILE)
    if actual != OKRU["sha256"]:
        raise ValueError(f"SHA-256 {OKRU_FILE.name} не совпал: {actual}")
    df = pd.read_json(OKRU_FILE, lines=True, dtype={"text": str})
    df.insert(0, "okru_row", range(len(df)))
    return df


def clean_external(okru: pd.DataFrame, ours: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Убирает тексты, совпадающие с нашим корпусом, и дубликаты внутри ok.ru."""
    okru = okru.assign(key=okru["text"].map(overlap_key))
    our_keys = set(ours[data.TEXT_COL].map(overlap_key))
    in_ours = okru["key"].isin(our_keys)
    empty = okru["key"] == ""
    labels_per_key = okru.groupby("key")["label"].nunique()
    conflicting = okru["key"].map(labels_per_key) > 1
    duplicate = okru.duplicated("key", keep="first")
    keep = ~(in_ours | empty | conflicting | duplicate)
    report = {
        "rows": int(len(okru)),
        "overlap_with_our_corpus": int(in_ours.sum()),
        "empty_after_normalization": int(empty.sum()),
        "conflicting_label_rows": int((conflicting & ~in_ours).sum()),
        "duplicate_rows_dropped": int((duplicate & ~in_ours & ~conflicting).sum()),
        "rows_kept": int(keep.sum()),
        "toxic_share_kept": round(float(okru.loc[keep, "label"].mean()), 4),
    }
    return okru[keep].drop(columns="key").reset_index(drop=True), report
