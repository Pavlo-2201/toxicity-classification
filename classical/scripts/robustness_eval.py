"""Нормализация и устойчивость к искажениям: word, char и word+char модели.

1. Варианты нормализации (`toxic_clf.robustness.NORMALIZATIONS`) сравниваются 5-fold CV
   на train: AP и F1 при пороге, подобранном на OOF. Так видно, чего вариант стоит на
   обычных текстах, и выбор не смотрит ни на validation, ни на test.
2. Набор устойчивости: тексты validation и их искажённые копии (латиница-двойник,
   маскировка, растянутые буквы, опечатки, всё вместе; seed фиксирован). Метки не
   меняются: искажения не меняют смысл. Для каждой модели (baseline и выбранная
   нормализация) — F1 при OOF-пороге, AP и доля текстов, у которых сменилось решение.
3. Test читается один раз, только для итоговой модели и её варианта с выбранной
   нормализацией: обычные тексты и те же искажения.

Тексты не сохраняются: набор генерируется заново из скачанного источника.
"""

from __future__ import annotations

from toxic_core.environment import capture

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from toxic_clf import data
from toxic_clf.models import build_pipeline
from toxic_clf.robustness import NORMALIZATIONS, PERTURBATIONS, perturb
from toxic_clf.selection import best_f1_threshold, out_of_fold_scores
from toxic_core import splits
from toxic_core.stdio import use_utf8_stdio

REPORTS = data.PROJECT_ROOT / "reports"
SEED = 2026

MODELS = {
    "word12_linearsvc_balanced": {"model": "linearsvc", "model_params": {"C": 1.0, "class_weight": "balanced"},
                                  "word": {"ngram_range": [1, 2]}},
    "charwb25_linearsvc_C0.3_balanced": {"model": "linearsvc", "model_params": {"C": 0.3, "class_weight": "balanced"},
                                         "char": {"analyzer": "char_wb", "ngram_range": [2, 5], "lowercase": True}},
    "word12+charwb25_linearsvc_balanced": {"model": "linearsvc", "model_params": {"C": 1.0, "class_weight": "balanced"},
                                           "word": {"ngram_range": [1, 2]},
                                           "char": {"analyzer": "char_wb", "ngram_range": [2, 5], "lowercase": True}},
}
FINAL = "charwb25_linearsvc_C0.3_balanced"
KINDS = ["clean", *PERTURBATIONS, "combo"]


def config_for(model: str, normalization: str) -> dict:
    return {**MODELS[model], "name": f"{model}|{normalization}", "normalization": normalization}


def perturbed_sets(texts: list[str], ids: list[int]) -> dict[str, list[str]]:
    sets = {"clean": texts}
    for kind in KINDS[1:]:
        # seed зависит от текста (source_row), а не от позиции: набор не меняется от порядка строк
        sets[kind] = [perturb(t, kind, SEED * 1_000_003 + i) for t, i in zip(texts, ids)]
    return sets


def evaluate(pipeline, threshold: float, sets: dict[str, list[str]], y) -> dict:
    clean_pred = None
    result = {}
    for kind, texts in sets.items():
        scores = pipeline.decision_function(texts)
        pred = (scores >= threshold).astype(int)
        if kind == "clean":
            clean_pred = pred
        result[kind] = {
            "f1_toxic": round(float(f1_score(y, pred)), 4),
            "average_precision": round(float(average_precision_score(y, scores)), 4),
            "flip_rate": round(float((pred != clean_pred).mean()), 4),
        }
    for kind in KINDS[1:]:
        result[kind]["f1_drop"] = round(result["clean"]["f1_toxic"] - result[kind]["f1_toxic"], 4)
    return result


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate-test", action="store_true", help="один раз оценить итоговую модель на test")
    args = parser.parse_args()

    parts = splits.load_splits()
    train, val, test = parts["train"], parts["validation"], parts["test"]
    x_train, y_train = train[data.TEXT_COL].tolist(), train[data.LABEL_COL].to_numpy()
    report = {"environment": capture(), "seed": SEED, "perturbations": {k: {"min_word_len": v[1], "word_share": v[2]} for k, v in PERTURBATIONS.items()},
              "normalization_cv": {}, "robustness_validation": {}}

    # 1. нормализация: CV на train
    thresholds = {}
    for model in MODELS:
        for norm in NORMALIZATIONS:
            config = config_for(model, norm)
            oof = out_of_fold_scores(config, x_train, y_train)
            threshold, f1 = best_f1_threshold(y_train, oof)
            thresholds[(model, norm)] = threshold
            report["normalization_cv"][config["name"]] = {
                "oof_f1_toxic_at_oof_threshold": round(float(f1), 4),
                "oof_average_precision": round(float(average_precision_score(y_train, oof)), 4),
                "oof_threshold": round(float(threshold), 6),
            }
            print(config["name"], report["normalization_cv"][config["name"]], flush=True)

    # допустимые варианты: на обычных текстах (OOF AP на train) теряют не больше 0.002 против baseline
    allowed = {}
    for model in MODELS:
        base = report["normalization_cv"][f"{model}|baseline"]["oof_average_precision"]
        allowed[model] = [n for n in NORMALIZATIONS
                          if report["normalization_cv"][f"{model}|{n}"]["oof_average_precision"] >= base - 0.002]
    report["allowed_normalizations"] = allowed

    # 2. устойчивость на validation: все варианты нормализации
    val_sets = perturbed_sets(val[data.TEXT_COL].tolist(), val[data.ID_COL].tolist())
    y_val = val[data.LABEL_COL].to_numpy()
    report["changed_texts_share_validation"] = {
        k: round(float(np.mean([a != b for a, b in zip(val_sets["clean"], val_sets[k])])), 4) for k in KINDS[1:]}
    for model in MODELS:
        for norm in NORMALIZATIONS:
            pipeline = build_pipeline(config_for(model, norm)).fit(x_train, y_train)
            key = f"{model}|{norm}"
            report["robustness_validation"][key] = evaluate(pipeline, thresholds[(model, norm)], val_sets, y_val)
            print(key, json.dumps(report["robustness_validation"][key], ensure_ascii=False), flush=True)

    # выбор для test: лучший F1 на искажениях combo (validation) среди допустимых вариантов
    chosen = {model: max(allowed[model], key=lambda n: report["robustness_validation"][f"{model}|{n}"]["combo"]["f1_toxic"])
              for model in MODELS}
    report["chosen_normalization"] = chosen
    report["choice_rule"] = ("среди вариантов, которые на обычных текстах (OOF AP train) не хуже baseline больше чем на "
                             "0.002, — лучший F1 на validation с искажениями combo")

    # 3. test: один раз, итоговая модель
    if args.evaluate_test:
        test_sets = perturbed_sets(test[data.TEXT_COL].tolist(), test[data.ID_COL].tolist())
        y_test = test[data.LABEL_COL].to_numpy()
        report["robustness_test"] = {}
        for norm in sorted({"baseline", chosen[FINAL]}):
            pipeline = build_pipeline(config_for(FINAL, norm)).fit(x_train, y_train)
            report["robustness_test"][f"{FINAL}|{norm}"] = evaluate(pipeline, thresholds[(FINAL, norm)], test_sets, y_test)
        print(json.dumps(report["robustness_test"], ensure_ascii=False, indent=1))

    (REPORTS / "robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
