"""Длины текстов и дубликаты."""

from __future__ import annotations

from toxic_clf.environment import capture

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toxic_clf import data, splits

REPORTS = data.PROJECT_ROOT / "reports"


def describe(values) -> dict:
    q = np.percentile(values, [5, 25, 50, 75, 95, 99])
    return {
        "mean": round(float(np.mean(values)), 1),
        "min": int(np.min(values)),
        **{f"p{p}": float(v) for p, v in zip([5, 25, 50, 75, 95, 99], q)},
        "max": int(np.max(values)),
    }


def main() -> None:
    df = data.load_dataset()
    text = df[data.TEXT_COL].str.strip()
    df = df.assign(n_chars=text.str.len(), n_words=text.str.split().str.len())

    lengths = {}
    for label, part in df.groupby(data.LABEL_COL):
        lengths[str(label)] = {"chars": describe(part["n_chars"]), "words": describe(part["n_words"])}
    short = {f"<= {n} chars": {str(k): int(v) for k, v in df[df.n_chars <= n][data.LABEL_COL].value_counts().sort_index().items()} for n in (30, 40, 60)}

    resolved = splits.resolve_duplicates(df)
    reasons = resolved["exclusion_reason"]
    dup_groups = resolved.groupby("group_sha256").size()
    duplicates = {
        "exact_duplicate_rows": int(df[data.TEXT_COL].duplicated().sum()),
        "rows_in_duplicate_groups": int(dup_groups[dup_groups > 1].sum()),
        "duplicate_groups": int((dup_groups > 1).sum()),
        "max_group_size": int(dup_groups.max()),
        "conflicting_groups": int(resolved[reasons == "conflicting_labels"]["group_sha256"].nunique()),
        "rows_excluded_conflicting_labels": int((reasons == "conflicting_labels").sum()),
        "rows_excluded_as_copies": int(reasons.str.startswith("duplicate_of_").sum()),
        "rows_kept": int((reasons == "").sum()),
        "note": "Дубликаты отличаются только хвостовым переводом строки; точных повторов строк нет. Минимальная длина текста в датасете 21 символ (видимо, фильтр при сборе).",
    }

    report = {"environment": capture(), "lengths_by_label": lengths, "short_texts_by_label": short, "duplicates": duplicates}
    (REPORTS / "text_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(0, np.log10(df.n_chars.max() + 1), 50)
    for label, name in [(0, "нетоксичные"), (1, "токсичные")]:
        part = df[df[data.LABEL_COL] == label]
        axes[0].hist(part.n_chars, bins=bins, alpha=0.6, label=name, density=True)
        axes[1].hist(part.n_words.clip(upper=100), bins=50, alpha=0.6, label=name, density=True)
    axes[0].set(xscale="log", xlabel="длина, символов (лог. шкала)", ylabel="плотность")
    axes[1].set(xlabel="длина, слов (обрезано на 100)", ylabel="плотность")
    for ax in axes:
        ax.legend()
    fig.suptitle("Распределение длины комментариев по классам")
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "length_by_label.png", dpi=110)


if __name__ == "__main__":
    main()
