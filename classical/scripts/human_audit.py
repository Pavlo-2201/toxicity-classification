"""Слепая человеческая перепроверка 120 текстов выборки разметки.

    python scripts/human_audit.py form       # reports/local/human_label_audit_form.html (с текстами, не публикуется)
    python scripts/human_audit.py compare    # после разметки: data/label_audit_human.csv -> отчёты

Форма открывается в браузере; ответы скачиваются кнопкой в CSV, который нужно положить в
data/label_audit_human.csv. Сравнение пишет reports/label_audit_human.json (сводка) и
reports/label_audit_discrepancies.csv (расхождения с LLM, без текстов), а версию с
текстами для разбора — в reports/local/.
"""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json
from pathlib import Path

import pandas as pd

from toxic_clf import data
from toxic_clf.human_audit import AUDIT_PATH, FORM_PATH, HUMAN_PATH, compare, load_human, wilson_interval, write_form
from toxic_core.stdio import use_utf8_stdio

REPORTS = data.PROJECT_ROOT / "reports"


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    form = sub.add_parser("form", help="сгенерировать слепую форму")
    form.add_argument("--output", type=Path, default=FORM_PATH)
    cmp = sub.add_parser("compare", help="сравнить ответы человека с датасетом и с LLM")
    cmp.add_argument("--human", type=Path, default=HUMAN_PATH)
    cmp.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = parser.parse_args()

    if args.command == "form":
        print(f"Форма: {write_form(args.output)}")
        return

    audit = pd.read_csv(AUDIT_PATH, keep_default_na=False)
    summary, discrepancies = compare(audit, load_human(args.human))
    summary["environment"] = capture()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    audit_summary = {"environment": capture()}
    for label, part in audit.groupby("toxic"):
        n = len(part)
        disagree = int((part["review"] == "disagree").sum())
        audit_summary[f"label_{label}"] = {"n": n, "disagree": disagree,
            "unclear": int((part["review"] == "unclear").sum()),
            "disagree_share": round(disagree / n, 3), "wilson_95": wilson_interval(disagree, n)}
    audit_summary["method"] = ("120 текстов train (по 60 каждой метки, seed 2026, без конфликтных дубликатов); "
                               "прочитаны моделью LLM по определению: оскорбление, угроза, непристойность, ненависть к группе")
    (args.reports_dir / "label_audit.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.reports_dir / "label_audit_human.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    discrepancies.to_csv(args.reports_dir / "label_audit_discrepancies.csv", index=False, lineterminator="\n")
    texts = data.load_dataset().set_index(data.ID_COL)[data.TEXT_COL]
    local = args.reports_dir / "local"
    local.mkdir(parents=True, exist_ok=True)
    discrepancies.assign(comment=discrepancies[data.ID_COL].map(texts)).to_csv(
        local / "label_audit_discrepancies_with_text.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"Расхождений с LLM: {len(discrepancies)}; с текстами — {local / 'label_audit_discrepancies_with_text.csv'}")


if __name__ == "__main__":
    main()
