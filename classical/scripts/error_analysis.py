"""Анализ ошибок лучшей модели на validation."""

from __future__ import annotations

from toxic_clf.environment import capture

import json
import re

import pandas as pd

from toxic_clf import data, splits
from toxic_clf.errors import categorize
from toxic_clf.models import build_pipeline

REPORTS = data.PROJECT_ROOT / "reports"
ANNOTATIONS = data.PROJECT_ROOT / "data" / "error_annotations.csv"
WORD = re.compile(r"\w+")


def main() -> None:
    best = json.loads((REPORTS / "best_model.json").read_text(encoding="utf-8"))
    config, threshold = best["best_config"], best["threshold"]["used"]

    parts = splits.load_splits()
    train, val = parts["train"], parts["validation"]
    pipeline = build_pipeline(config).fit(train[data.TEXT_COL], train[data.LABEL_COL])

    train_vocab = {w for t in train[data.TEXT_COL] for w in WORD.findall(t.lower())}

    def oov_share(text: str) -> float:
        words = WORD.findall(text.lower())
        return sum(w not in train_vocab for w in words) / max(len(words), 1)

    val = val.copy()
    val["score"] = pipeline.decision_function(val[data.TEXT_COL])
    val["pred"] = (val["score"] >= threshold).astype(int)
    val["oov_share"] = val[data.TEXT_COL].map(oov_share).round(3)
    cats = pd.DataFrame([categorize(t, o) for t, o in zip(val[data.TEXT_COL], val["oov_share"])], index=val.index)
    val = pd.concat([val, cats], axis=1)
    val["error"] = ""
    val.loc[(val.pred == 1) & (val[data.LABEL_COL] == 0), "error"] = "FP"
    val.loc[(val.pred == 0) & (val[data.LABEL_COL] == 1), "error"] = "FN"

    groups = {}
    for name in cats.columns:
        part = val[val[name]]
        neg, pos = part[part[data.LABEL_COL] == 0], part[part[data.LABEL_COL] == 1]
        groups[name] = {
            "n": int(len(part)),
            "n_toxic": int(len(pos)),
            "fp_rate": round(float((neg.pred == 1).mean()), 3) if len(neg) else None,
            "fn_rate": round(float((pos.pred == 0).mean()), 3) if len(pos) else None,
            "error_rate": round(float((part.pred != part[data.LABEL_COL]).mean()), 3) if len(part) else None,
        }
    neg, pos = val[val[data.LABEL_COL] == 0], val[val[data.LABEL_COL] == 1]
    groups["all"] = {
        "n": int(len(val)),
        "n_toxic": int(len(pos)),
        "fp_rate": round(float((neg.pred == 1).mean()), 3),
        "fn_rate": round(float((pos.pred == 0).mean()), 3),
        "error_rate": round(float((val.pred != val[data.LABEL_COL]).mean()), 3),
    }
    report = {
        "environment": capture(),
        "model": config["name"],
        "threshold": threshold,
        "split": "validation",
        "n_fp": int((val.error == "FP").sum()),
        "n_fn": int((val.error == "FN").sum()),
        "groups": groups,
        "group_definitions": {
            "short": "<= 40 символов",
            "obscene": "есть корень обсценной лексики",
            "insult_lexicon": "есть явное оскорбление из списка",
            "obfuscation_or_typos": "маскировка (*, точки, смесь алфавитов, растянутые буквы) или >= 30% слов вне словаря train",
            "no_explicit_markers": "нет ни мата, ни явных оскорблений",
        },
    }
    (REPORTS / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = [data.ID_COL, "error", data.LABEL_COL, "pred", "score", "oov_share", *cats.columns, data.TEXT_COL]
    errors = val[val.error != ""].sort_values("score", key=abs, ascending=False)[cols]
    errors.assign(score=errors.score.round(3)).to_csv(REPORTS / "validation_errors.csv", index=False)

    if ANNOTATIONS.exists():
        notes = pd.read_csv(ANNOTATIONS, keep_default_na=False)
        examples = notes.merge(val, on=data.ID_COL, how="left", validate="one_to_one")
        if examples["error"].isna().any() or (examples["error"] == "").any():
            raise SystemExit("В аннотациях есть строки, которые не являются ошибками модели на validation")
        examples = examples[[data.ID_COL, "error", data.LABEL_COL, "score", "category", "explanation", data.TEXT_COL]]
        examples.assign(score=examples.score.round(3)).to_csv(REPORTS / "error_examples.csv", index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
