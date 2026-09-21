"""Пересчёт эвристических групп ошибок по сохранённым предсказаниям, без моделей.

final_comparison.py считает группы вместе с инференсом трёх моделей. Когда меняются только
правила групп (src/rubert_clf/errors.py), достаточно предсказаний из
reports/validation_predictions_all.csv: скрипт обновляет в нём столбцы групп и раздел by_group
в reports/error_analysis_rubert.json. Остальные поля отчёта не меняются.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rubert_clf import data
from rubert_clf.errors import categorize, oov_share, rates_by_group, train_vocabulary
from rubert_clf.finetune import write_json
from toxic_core import splits
from toxic_core.stdio import use_utf8_stdio

REPORTS = data.PROJECT_ROOT / "reports"
PREDICTIONS = REPORTS / "validation_predictions_all.csv"
ANALYSIS = REPORTS / "error_analysis_rubert.json"
MODELS = {"tfidf_linearsvc": "tfidf_pred", "frozen_rubert_logreg": "frozen_pred", "finetuned_rubert": "finetuned_pred"}


def main() -> None:
    use_utf8_stdio()
    parts = splits.load_splits()
    val = parts["validation"]
    table = pd.read_csv(PREDICTIONS)
    if not (table[data.ID_COL].to_numpy() == val[data.ID_COL].to_numpy()).all():
        raise SystemExit(f"Порядок строк {PREDICTIONS.name} не совпадает с validation")
    if not (table[data.LABEL_COL].to_numpy() == val[data.LABEL_COL].to_numpy()).all():
        raise SystemExit(f"Метки в {PREDICTIONS.name} не совпадают с validation")

    vocab = train_vocabulary(parts["train"][data.TEXT_COL])
    cats = pd.DataFrame([categorize(t, oov_share(t, vocab)) for t in val[data.TEXT_COL]])
    for c in cats.columns:
        table[c] = cats[c].to_numpy()
    table.to_csv(PREDICTIONS, index=False, lineterminator="\n")

    y = table[data.LABEL_COL].to_numpy()
    groups = {"all": np.ones(len(table), bool), **{c: cats[c].to_numpy() for c in cats.columns}}
    report = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    report["by_group"] = rates_by_group(groups, y, {m: table[col].to_numpy() for m, col in MODELS.items()})
    write_json(ANALYSIS, report)
    print(json.dumps(report["by_group"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
