import json

import pandas as pd
import pytest

from toxic_clf.human_audit import compare, form_items, load_human, render_form, wilson_interval

AUDIT = pd.DataFrame({
    "source_row": [10, 11, 12, 13, 20, 21, 22, 23],
    "toxic": [0, 0, 0, 0, 1, 1, 1, 1],
    "review": ["agree", "agree", "disagree", "agree", "agree", "disagree", "unclear", "agree"],
    "note": ["", "", "выпад", "", "", "лозунг", "", ""],
})
DATASET = pd.DataFrame({"source_row": AUDIT["source_row"], "comment": [f"текст {r}" for r in AUDIT["source_row"]],
                        "toxic": AUDIT["toxic"]})


def test_wilson_matches_published_intervals():
    assert wilson_interval(1, 60) == [0.003, 0.089]
    assert wilson_interval(8, 60) == [0.069, 0.242]


def test_form_is_blind_and_shuffled_deterministically():
    items = form_items(AUDIT, DATASET)
    assert all(set(item) == {"row", "text"} for item in items)
    assert sorted(item["row"] for item in items) == sorted(AUDIT["source_row"])
    assert [item["row"] for item in items] != sorted(AUDIT["source_row"])
    assert items == form_items(AUDIT, DATASET)
    page = render_form(items)
    for hidden in ("agree", "disagree", "выпад", "лозунг", '"toxic"'):
        assert hidden not in page
    embedded = json.loads(page.split("const ITEMS = ", 1)[1].split(";\n", 1)[0])
    assert embedded == items


def test_compare_counts_against_dataset_and_llm():
    human = pd.DataFrame({
        "source_row": [10, 11, 12, 13, 20, 21, 22, 23],
        "human_label": ["0", "1", "1", "0", "1", "1", "0", "unclear"],
        "human_note": [""] * 8,
    })
    summary, discrepancies = compare(AUDIT, human)
    assert summary["human_vs_dataset"]["label_0"]["disagree"] == 2  # строки 11 и 12
    assert summary["human_vs_dataset"]["label_1"]["disagree"] == 1 and summary["human_vs_dataset"]["label_1"]["unclear"] == 1
    llm = summary["human_vs_llm"]
    # LLM: 10→0, 11→0, 12→1, 13→0, 20→1, 21→0, 22→unclear, 23→1
    assert llm["both_decisive"] == 6 and llm["agreement"] == round(4 / 6, 3)
    assert llm["confusion"]["human_1_llm_0"] == 2
    assert discrepancies["source_row"].tolist() == [11, 21, 22, 23]
    assert "comment" not in discrepancies.columns


def test_compare_requires_the_same_sample():
    human = pd.DataFrame({"source_row": [10], "human_label": ["0"], "human_note": [""]})
    with pytest.raises(ValueError, match="не совпадают"):
        compare(AUDIT, human)


def test_load_human_rejects_unknown_labels(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("source_row,human_label,human_note\n10,да,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="да"):
        load_human(path)
