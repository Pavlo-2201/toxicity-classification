"""Итоговая таблица экспериментов из reports/experiments/*.json."""

from __future__ import annotations

import json

import pandas as pd

from toxic_clf import data
from toxic_core.stdio import use_utf8_stdio

REPORTS = data.PROJECT_ROOT / "reports"


def main() -> None:
    use_utf8_stdio()
    rows = []
    for path in sorted((REPORTS / "experiments").glob("stage*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload["results"]:
            if "metrics" in r:
                m = r["metrics"]
                rows.append({"stage": payload["stage"], "name": r["name"], "evaluated_on": "validation",
                             "f1_toxic": m["f1_toxic"], "f1_toxic_std": None, "macro_f1": m["macro_f1"],
                             "precision_toxic": m["precision_toxic"], "recall_toxic": m["recall_toxic"],
                             "average_precision": m.get("average_precision"), "n_features": r.get("n_features")})
            else:
                cv = r["cv"]
                rows.append({"stage": payload["stage"], "name": r["name"], "evaluated_on": "train 5-fold CV",
                             "f1_toxic": cv["f1_toxic"]["mean"], "f1_toxic_std": cv["f1_toxic"]["std"],
                             "macro_f1": cv["macro_f1"]["mean"], "precision_toxic": cv["precision_toxic"]["mean"],
                             "recall_toxic": cv["recall_toxic"]["mean"], "average_precision": None, "n_features": None})

    final = json.loads((REPORTS / "final_metrics.json").read_text(encoding="utf-8"))
    for split in ("validation", "test"):
        m = final[split]
        rows.append({"stage": "final", "name": f"{final['model']} (порог {final['threshold']:.4f})", "evaluated_on": split,
                     "f1_toxic": m["f1_toxic"], "f1_toxic_std": None, "macro_f1": m["macro_f1"],
                     "precision_toxic": m["precision_toxic"], "recall_toxic": m["recall_toxic"],
                     "average_precision": m.get("average_precision"), "n_features": None})

    environment = final["environment"]
    table = pd.DataFrame(rows).assign(python=environment["python"], sklearn=environment["sklearn"])
    table.to_csv(REPORTS / "experiments_summary.csv", index=False, lineterminator="\n")
    md = [f"Окружение: Python {environment['python']}, scikit-learn {environment['sklearn']}, "
          f"NumPy {environment['numpy']}, SciPy {environment['scipy']}, pandas {environment['pandas']}." , "", "| этап | конфигурация | оценка на | F1 toxic | ± std | macro-F1 | precision | recall | AP |", "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None or pd.isna(v) else f"{v:.4f}"
    for r in table.itertuples():
        md.append(f"| {r.stage} | {r.name} | {r.evaluated_on} | {fmt(r.f1_toxic)} | {fmt(r.f1_toxic_std)} | {fmt(r.macro_f1)} | "
                  f"{fmt(r.precision_toxic)} | {fmt(r.recall_toxic)} | {fmt(r.average_precision)} |")
    (REPORTS / "experiments_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(table)} строк -> reports/experiments_summary.csv, reports/experiments_summary.md")


if __name__ == "__main__":
    main()
