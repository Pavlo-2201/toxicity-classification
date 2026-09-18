"""Сводка экспериментов: среднее и разброс по seed, выбор по правилу протокола, сравнение с TF-IDF.

Читает <каталог>/runs/*.json, пишет в тот же каталог summary.json, summary.md и анализ ошибок
итоговой конфигурации на test (error_groups_test.json, без текстов). Каталог по умолчанию —
опубликованные результаты experiments/runs; для нового запуска — --run-dir experiments/runs/<id>: такой
каталог сначала проверяется (scripts/check_experiment_run.py), неполный запуск не сводится.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from rubert_clf import data, experiment_run as er, splits
from rubert_clf.cli import use_utf8_stdio
from rubert_clf.errors import categorize, oov_share, rates_by_group, train_vocabulary
from rubert_clf.finetune import write_json
from rubert_clf.metrics import classification_metrics, mcnemar, paired_bootstrap_f1_diff
from rubert_clf.protocol import long_mask

ROOT = data.PROJECT_ROOT
PUBLISHED = ROOT / "experiments" / "runs"
PROTOCOL = ROOT / "configs" / "protocol.json"
CLASSICAL = ROOT / "reference" / "classical" / "test_predictions.csv"


def stat(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "runs": [round(float(x), 4) for x in v]}


def summarize_candidate(runs: list[dict]) -> dict:
    g = lambda f: [f(r) for r in runs]
    first = runs[0]
    return {
        "seeds": g(lambda r: r["spec"]["seed"]),
        "selected_epochs": g(lambda r: r["selected_epoch"]),
        "thresholds": g(lambda r: round(r["threshold"], 4)),
        "validation_ap": stat(g(lambda r: r["validation"]["at_0.5"]["average_precision"])),
        "validation_long_ap": stat(g(lambda r: r["validation"]["long_texts"]["average_precision"])),
        "test_ap": stat(g(lambda r: r["test"]["at_0.5"]["average_precision"])),
        "test_f1_at_0.5": stat(g(lambda r: r["test"]["at_0.5"]["f1_toxic"])),
        "test_f1_at_calibrated_threshold": stat(g(lambda r: r["test"]["at_calibrated_threshold"]["f1_toxic"])),
        "test_long_ap": stat(g(lambda r: r["test"]["long_texts"]["average_precision"])),
        "test_long_f1": stat(g(lambda r: r["test"]["long_texts"]["f1_at_calibrated_threshold"])),
        "trainable_parameters": first["parameters"]["trainable"],
        "total_parameters": first["parameters"]["total"],
        "gpu_peak_memory_mb": max(g(lambda r: r["gpu_peak_memory_mb"] or 0)),
        "train_seconds_per_epoch_gpu": stat(g(lambda r: np.mean([e["train_seconds"] for e in r["epochs"]]))),
        "train_texts_per_second_gpu": stat(g(lambda r: np.mean([e["train_texts_per_second"] for e in r["epochs"]]))),
        "gpu_batch_ms_per_1000": stat(g(lambda r: r["inference"]["batch_ms_per_1000_texts"])),
        "cpu_single_text_ms_p50": next((r["inference_cpu"]["single_text_ms_p50"] for r in runs if "inference_cpu" in r), None),
        "cpu_batch_ms_per_1000": next((r["inference_cpu"]["batch_ms_per_1000_texts"] for r in runs if "inference_cpu" in r), None),
        # LoRA после слияния адаптера с весами (protocol.measure_merged); в выборе не участвует
        "cpu_single_text_ms_p50_merged": next((r["inference_cpu_merged"]["single_text_ms_p50"] for r in runs
                                               if "inference_cpu_merged" in r), None),
        "cpu_batch_ms_per_1000_merged": next((r["inference_cpu_merged"]["batch_ms_per_1000_texts"] for r in runs
                                              if "inference_cpu_merged" in r), None),
        "truncated_share_test": first["truncated_share"]["test"],
    }


def cpu_cell(s: dict) -> str:
    merged = s.get("cpu_single_text_ms_p50_merged")
    return f"{s['cpu_single_text_ms_p50']}" + (f" (после слияния {merged})" if merged is not None else "")


def select(summary: dict) -> dict:
    """Правило из configs/protocol.json (записано до запуска)."""
    best = max(summary, key=lambda n: summary[n]["validation_ap"]["mean"])
    margin = max(summary[best]["validation_ap"]["std"], 0.002)
    equivalent = [n for n in summary if summary[best]["validation_ap"]["mean"] - summary[n]["validation_ap"]["mean"] <= margin]
    by_cost = sorted(equivalent, key=lambda n: summary[n]["cpu_single_text_ms_p50"] or float("inf"))
    best_long = summary[best]["validation_long_ap"]["mean"]
    chosen = next(n for n in by_cost if summary[n]["validation_long_ap"]["mean"] >= best_long - 0.01)
    return {"best_by_validation_ap": best, "margin": margin, "equivalent": equivalent, "by_cpu_cost": by_cost,
            "chosen": chosen}


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=PUBLISHED, help="каталог с runs/ (по умолчанию experiments/runs)")
    args = parser.parse_args()
    out = args.run_dir
    protocol_path = PROTOCOL
    if (out / er.MANIFEST).exists():  # каталог experiments.py: только полный согласованный запуск
        problems = er.verify(out)
        if problems:
            raise SystemExit("Запуск не прошёл проверку:\n  " + "\n  ".join(problems))
        protocol_path = ROOT / er.read_json(out / er.MANIFEST)["fingerprint"]["protocol"]["path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    runs = defaultdict(list)
    ready = {}
    for path in sorted((out / er.RUNS).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if path.name.startswith("ready__"):
            ready[r["spec"]["name"]] = r
        else:
            runs[path.stem.split("__s")[0]].append(r)
    order = [c["name"] for c in protocol["candidates"] if c["name"] in runs]
    summary = {name: summarize_candidate(runs[name]) for name in order}
    selection = select(summary)

    test = splits.load_splits()["test"]
    y = test[data.LABEL_COL].to_numpy()
    long = long_mask(test[data.TEXT_COL])
    classical = pd.read_csv(CLASSICAL).set_index(data.ID_COL).loc[test[data.ID_COL]]
    tfidf_pred, tfidf_score = classical["pred"].to_numpy(), classical["score"].to_numpy()
    tfidf = {"test_ap": round(float(average_precision_score(y, tfidf_score)), 4),
             "test_f1_at_tuned_threshold": classification_metrics(y, tfidf_pred, tfidf_score)["f1_toxic"],
             "test_long_ap": round(float(average_precision_score(y[long], tfidf_score[long])), 4),
             "note": "порог подобран по OOF на train; F1 при 0.5 не определён: score — margin LinearSVC"}

    chosen = selection["chosen"]
    comparisons = []
    for r in runs[chosen]:
        probs = np.asarray(r["predictions"]["test"])
        pred = (probs >= r["threshold"]).astype(int)
        comparisons.append({"seed": r["spec"]["seed"], "mcnemar_vs_tfidf": mcnemar(y, pred, tfidf_pred),
                            "f1_diff_vs_tfidf_bootstrap": paired_bootstrap_f1_diff(y, pred, tfidf_pred)})
    diffs = [c["f1_diff_vs_tfidf_bootstrap"]["mean"] for c in comparisons]
    seed_vs_bootstrap = {
        "f1_diff_mean_over_seeds": round(float(np.mean(diffs)), 4),
        "f1_diff_std_over_seeds": round(float(np.std(diffs, ddof=1)), 4) if len(diffs) > 1 else 0.0,
        "bootstrap_ci_half_width_mean": round(float(np.mean([(c["f1_diff_vs_tfidf_bootstrap"]["high"] -
                                                              c["f1_diff_vs_tfidf_bootstrap"]["low"]) / 2 for c in comparisons])), 4),
        "note": "bootstrap оценивает разброс от выборки test при фиксированной модели, std по seed — от обучения",
    }

    # анализ ошибок итоговой конфигурации на test: test не участвовал ни в выборе эпохи, ни в выборе модели
    parts = splits.load_splits()
    vocab = train_vocabulary(parts["train"][data.TEXT_COL])
    cats = pd.DataFrame([categorize(t, oov_share(t, vocab)) for t in test[data.TEXT_COL]])
    votes = np.mean([(np.asarray(r["predictions"]["test"]) >= r["threshold"]).astype(int) for r in runs[chosen]], axis=0)
    majority = (votes >= 0.5).astype(int)
    groups = {"all": np.ones(len(test), bool), "long_texts": long, **{c: cats[c].to_numpy() for c in cats.columns}}
    error_groups = rates_by_group(groups, y, {"tfidf_linearsvc": tfidf_pred, f"{chosen}_majority_of_seeds": majority})
    unstable = int(((votes > 0) & (votes < 1)).sum())
    write_json(out / "error_groups_test.json", {"model": chosen, "split": "test", "by_group": error_groups,
                                               "texts_with_seed_disagreement": unstable,
                                               "note": "решение итоговой конфигурации — большинство по seed"})
    pd.DataFrame({data.ID_COL: test[data.ID_COL], data.LABEL_COL: y, "tfidf_pred": tfidf_pred,
                  "rubert_vote_share": votes.round(3), "rubert_majority_pred": majority, "long_text": long,
                  **{c: cats[c].to_numpy() for c in cats.columns}}).to_csv(out / "test_predictions.csv", index=False,
                                                                           lineterminator="\n")

    report = {"candidates": summary, "selection": selection, "tfidf_baseline": tfidf,
              "ready_classifiers": {n: {"test_ap": r["test"]["at_0.5"]["average_precision"],
                                        "test_f1_at_0.5": r["test"]["at_0.5"]["f1_toxic"],
                                        "test_f1_at_calibrated_threshold": r["test"]["at_calibrated_threshold"]["f1_toxic"],
                                        "test_long_ap": r["test"].get("long_texts", {}).get("average_precision"),
                                        "contamination": r["contamination"], "training_data": r["spec"]["training_data"]}
                                    for n, r in ready.items()},
              "chosen_vs_tfidf": comparisons, "seed_vs_bootstrap": seed_vs_bootstrap}
    write_json(out / "summary.json", report)

    f = lambda s: f"{s['mean']:.3f} ± {s['std']:.3f}"
    lines = ["| конфигурация | val AP | test AP | test F1 @0.5 | test F1 @порог калибровки | test AP длинные | CPU мс/текст | GPU с/эпоха | обучаемых параметров |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        mark = " **(выбор)**" if n == chosen else ""
        lines.append(f"| `{n}`{mark} | {f(s['validation_ap'])} | {f(s['test_ap'])} | {f(s['test_f1_at_0.5'])} | "
                     f"{f(s['test_f1_at_calibrated_threshold'])} | {f(s['test_long_ap'])} | {cpu_cell(s)} | "
                     f"{s['train_seconds_per_epoch_gpu']['mean']:.1f} | {s['trainable_parameters'] / 1e6:.1f} M |")
    lines.append(f"| TF-IDF + LinearSVC | — | {tfidf['test_ap']:.3f} | — | {tfidf['test_f1_at_tuned_threshold']:.3f} | {tfidf['test_long_ap']:.3f} | | | |")
    for n, r in report["ready_classifiers"].items():
        lines.append(f"| `{n}` (видел данные задачи или ok.ru) | — | {r['test_ap']:.3f} | {r['test_f1_at_0.5']:.3f} | "
                     f"{r['test_f1_at_calibrated_threshold']:.3f} | {r['test_long_ap']} | | | |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(json.dumps(selection, ensure_ascii=False))
    print(json.dumps(seed_vs_bootstrap, ensure_ascii=False))


if __name__ == "__main__":
    main()
