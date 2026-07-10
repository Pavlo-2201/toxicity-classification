"""Единственный источник разбиений train/validation/test для обеих моделей.

Манифест `data/splits/split_manifest.csv` — по строке на каждую исходную запись: хеш текста,
ключ дедупликации, метка, разбиение и причина исключения, без самих текстов. Его SHA-256
закреплён здесь и в `data/splits/data_fingerprint.json`; `load_manifest` проверяет его при каждом
чтении, `load_splits` — хеши текстов и метки датасета построчно.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from toxic_core import data

SPLITS_DIR = data.DATA_DIR / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.csv"
FINGERPRINT_PATH = SPLITS_DIR / "data_fingerprint.json"
MANIFEST_SHA256 = "a5c21437095942fde727b0775624660d0d083cd411348594c449305a704b23de"

SPLIT_SEED = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
SPLITS = ("train", "validation", "test")
EXCLUDED = "excluded"

_WHITESPACE = re.compile(r"\s+")


def duplicate_key(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет group_sha256 и exclusion_reason ('' — строка остаётся)."""
    out = df[[data.ID_COL, data.TEXT_COL, data.LABEL_COL]].copy()
    keys = out[data.TEXT_COL].map(duplicate_key)
    out["group_sha256"] = keys.map(lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    out = out.sort_values(data.ID_COL).reset_index(drop=True)

    groups = out.groupby("group_sha256")
    n_labels = groups[data.LABEL_COL].transform("nunique")
    first_row = groups[data.ID_COL].transform("min")

    reason = pd.Series("", index=out.index)
    is_copy = out[data.ID_COL] != first_row
    reason[is_copy] = "duplicate_of_" + first_row[is_copy].astype(str)
    reason[n_labels > 1] = "conflicting_labels"
    out["exclusion_reason"] = reason
    return out


def assign_splits(resolved: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    kept = resolved[resolved["exclusion_reason"] == ""]
    train_val, test = train_test_split(
        kept, test_size=TEST_SIZE, stratify=kept[data.LABEL_COL], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val[data.LABEL_COL],
        random_state=seed,
    )
    split = pd.Series(EXCLUDED, index=resolved.index)
    split[train.index] = "train"
    split[val.index] = "validation"
    split[test.index] = "test"
    return resolved.assign(split=split)


def build_manifest(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Манифест: по строке на каждую исходную запись, без самих текстов."""
    assigned = assign_splits(resolve_duplicates(df), seed=seed)
    manifest = pd.DataFrame(
        {
            data.ID_COL: assigned[data.ID_COL],
            "text_sha256": assigned[data.TEXT_COL].map(data.text_sha256),
            "group_sha256": assigned["group_sha256"],
            data.LABEL_COL: assigned[data.LABEL_COL],
            "split": assigned["split"],
            "exclusion_reason": assigned["exclusion_reason"],
        }
    )
    return manifest.sort_values(data.ID_COL).reset_index(drop=True)


def load_manifest(path: Path = MANIFEST_PATH, check_sha: bool = True) -> pd.DataFrame:
    if check_sha:
        actual = data.sha256_file(path)
        if actual != MANIFEST_SHA256:
            raise ValueError(f"SHA-256 манифеста {actual} не совпадает с зафиксированным {MANIFEST_SHA256}")
    return pd.read_csv(path, dtype={"exclusion_reason": str}, keep_default_na=False)


def load_fingerprint(path: Path = FINGERPRINT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_splits(dataset: pd.DataFrame | None = None, manifest: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """{'train', 'validation', 'test'} -> DataFrame[source_row, comment, toxic], сверяя хеши текстов с манифестом."""
    dataset = data.load_dataset() if dataset is None else dataset
    manifest = load_manifest() if manifest is None else manifest
    merged = manifest.merge(dataset, on=data.ID_COL, how="left", suffixes=("", "_dataset"), validate="one_to_one")
    if merged[data.TEXT_COL].isna().any():
        raise ValueError("В датасете нет части строк манифеста")
    if not (merged[data.TEXT_COL].map(data.text_sha256) == merged["text_sha256"]).all():
        raise ValueError("Хеши текстов не совпадают с манифестом: данные отличаются")
    if not (merged[data.LABEL_COL] == merged[f"{data.LABEL_COL}_dataset"]).all():
        raise ValueError("Метки не совпадают с манифестом")
    cols = [data.ID_COL, data.TEXT_COL, data.LABEL_COL]
    return {name: merged.loc[merged["split"] == name, cols].reset_index(drop=True) for name in SPLITS}


def split_counts(manifest: pd.DataFrame, names=(*SPLITS, EXCLUDED)) -> dict:
    counts = {}
    for name in names:
        part = manifest[manifest["split"] == name]
        counts[name] = {
            "rows": int(len(part)),
            "label_counts": {str(k): int(v) for k, v in part[data.LABEL_COL].value_counts().sort_index().items()},
        }
    return counts


def fingerprint_info(df: pd.DataFrame, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Содержимое data_fingerprint.json: происхождение данных, правило разбиения и SHA-256 манифеста."""
    return {
        "dataset": data.ORIGINAL_DATASET,
        "source": {
            "repo": data.SOURCE_REPO,
            "commit": data.SOURCE_COMMIT,
            "path": data.SOURCE_PATH,
            "url": data.SOURCE_URL,
            "file_sha256": data.SOURCE_SHA256,
            "subset_rows": [data.SUBSET_START_ROW, data.SUBSET_START_ROW + data.EXPECTED_ROWS - 1],
        },
        "content_fingerprint": data.dataset_fingerprint(df),
        "content_fingerprint_definition": "sha256 по строкам 'source_row\\tsha256(text utf-8)\\tlabel\\n' в порядке source_row",
        "split": {
            "method": "stratified train_test_split по уникальным текстам",
            "seed": SPLIT_SEED,
            "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE,
            "duplicate_key": "NFC + схлопывание пробелов + strip + casefold",
            "conflicting_labels_rule": "группа одинаковых текстов с разными метками исключается целиком",
            "duplicates_rule": "из группы с одинаковой меткой остаётся запись с наименьшим source_row",
        },
        "counts": split_counts(load_manifest(manifest_path, check_sha=False)),
        "manifest_file": data.relative_to(manifest_path),
        "manifest_sha256": data.sha256_file(manifest_path),
    }


def compare_manifests(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Построчное сравнение двух манифестов. Возвращает отчёт с числом расхождений по колонкам."""
    cols = list(expected.columns)
    e = expected[cols].sort_values(data.ID_COL).reset_index(drop=True)
    a = actual[cols].sort_values(data.ID_COL).reset_index(drop=True)
    if len(e) != len(a) or not (e[data.ID_COL].values == a[data.ID_COL].values).all():
        return {"identical": False, "reason": "разный набор source_row", "rows_expected": len(e), "rows_actual": len(a)}
    mismatches = {c: int((e[c].astype(str).values != a[c].astype(str).values).sum()) for c in cols}
    return {"identical": all(v == 0 for v in mismatches.values()), "rows": len(e), "mismatches_by_column": mismatches}


def split_overlaps(manifest: pd.DataFrame) -> dict:
    """Число общих текстов (по SHA-256 текста и по ключу дедупликации) для каждой пары разбиений."""
    used = manifest[manifest["split"].isin(SPLITS)]
    result = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            for col in ("text_sha256", "group_sha256"):
                common = set(used.loc[used.split == a, col]) & set(used.loc[used.split == b, col])
                result[f"{a}-{b}:{col}"] = len(common)
    return result


def verify(dataset: pd.DataFrame | None = None) -> dict:
    """Полная проверка данных и разбиений: исходный файл, отпечаток, SHA-256 и пересборка манифеста,
    размеры разбиений и отсутствие общих текстов. Отчёт без путей машины и времени — детерминирован."""
    fingerprint = load_fingerprint()
    report: dict = {}
    report["source_file_sha256"] = data.sha256_file(data.SOURCE_FILE)
    report["source_file_sha256_ok"] = report["source_file_sha256"] == fingerprint["source"]["file_sha256"]

    dataset = data.load_dataset() if dataset is None else dataset
    report["content_fingerprint"] = data.dataset_fingerprint(dataset)
    report["content_fingerprint_ok"] = report["content_fingerprint"] == fingerprint["content_fingerprint"]

    manifest = load_manifest()  # проверяет SHA-256 манифеста
    report["manifest_sha256"] = MANIFEST_SHA256
    report["manifest_sha256_ok"] = MANIFEST_SHA256 == fingerprint["manifest_sha256"]
    report["rebuilt_vs_manifest"] = compare_manifests(manifest, build_manifest(dataset))

    parts = load_splits(dataset, manifest)
    report["split_counts"] = {
        name: {"rows": len(df), "label_counts": {str(k): int(v) for k, v in df[data.LABEL_COL].value_counts().sort_index().items()}}
        for name, df in parts.items()
    }
    report["split_counts_ok"] = report["split_counts"] == {name: fingerprint["counts"][name] for name in SPLITS}
    report["overlaps"] = split_overlaps(manifest)
    report["no_overlaps"] = all(v == 0 for v in report["overlaps"].values())
    report["passed"] = bool(all([
        report["source_file_sha256_ok"], report["content_fingerprint_ok"], report["manifest_sha256_ok"],
        report["rebuilt_vs_manifest"]["identical"], report["split_counts_ok"], report["no_overlaps"],
    ]))
    return report
