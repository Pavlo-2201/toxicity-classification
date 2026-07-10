"""Проверка меток, пустых текстов и дисбаланса классов."""

from __future__ import annotations

import json

from toxic_clf import data

REPORT_PATH = data.PROJECT_ROOT / "reports" / "data_check.json"


def main() -> None:
    df = data.load_dataset()
    report = {
        "dataset": data.ORIGINAL_DATASET,
        "source": {
            "repo": data.SOURCE_REPO,
            "commit": data.SOURCE_COMMIT,
            "path": data.SOURCE_PATH,
            "sha256": data.SOURCE_SHA256,
            "subset_start_row": data.SUBSET_START_ROW,
        },
        "fingerprint": data.dataset_fingerprint(df),
        "checks": data.check_dataset(df),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    if not report["checks"]["passed"]:
        raise SystemExit("Проверки не пройдены")


if __name__ == "__main__":
    main()
