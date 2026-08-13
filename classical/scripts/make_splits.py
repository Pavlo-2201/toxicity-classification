"""Манифест разбиений и отпечаток исходных данных."""

from __future__ import annotations

import json

from toxic_clf import data, splits
from toxic_clf.cli import use_utf8_stdio


def main() -> None:
    use_utf8_stdio()
    df = data.load_dataset()
    fingerprint = data.dataset_fingerprint(df)
    if fingerprint != data.EXPECTED_FINGERPRINT:
        raise SystemExit(f"Данные отличаются от ожидаемых: {fingerprint}")

    manifest = splits.build_manifest(df)
    splits.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(splits.MANIFEST_PATH, index=False, lineterminator="\n")

    counts = {}
    for name in (*splits.SPLITS, splits.EXCLUDED):
        part = manifest[manifest["split"] == name]
        counts[name] = {
            "rows": int(len(part)),
            "label_counts": {str(k): int(v) for k, v in part[data.LABEL_COL].value_counts().sort_index().items()},
        }
    info = {
        "dataset": data.ORIGINAL_DATASET,
        "source": {
            "repo": data.SOURCE_REPO,
            "commit": data.SOURCE_COMMIT,
            "path": data.SOURCE_PATH,
            "url": data.SOURCE_URL,
            "file_sha256": data.SOURCE_SHA256,
            "subset_rows": [data.SUBSET_START_ROW, data.SUBSET_START_ROW + data.EXPECTED_ROWS - 1],
        },
        "content_fingerprint": fingerprint,
        "content_fingerprint_definition": "sha256 по строкам 'source_row\\tsha256(text utf-8)\\tlabel\\n' в порядке source_row",
        "split": {
            "method": "stratified train_test_split по уникальным текстам",
            "seed": splits.SPLIT_SEED,
            "validation_size": splits.VALIDATION_SIZE,
            "test_size": splits.TEST_SIZE,
            "duplicate_key": "NFC + схлопывание пробелов + strip + casefold",
            "conflicting_labels_rule": "группа одинаковых текстов с разными метками исключается целиком",
            "duplicates_rule": "из группы с одинаковой меткой остаётся запись с наименьшим source_row",
        },
        "counts": counts,
        "manifest_file": splits.MANIFEST_PATH.relative_to(data.PROJECT_ROOT).as_posix(),
        "manifest_sha256": data.sha256_file(splits.MANIFEST_PATH),
    }
    splits.FINGERPRINT_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
