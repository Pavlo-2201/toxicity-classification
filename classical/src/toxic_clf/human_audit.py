"""Слепая человеческая перепроверка выборки разметки (120 текстов из data/label_audit.csv).

Прежнюю проверку выполнила модель LLM: она видела метку датасета и выносила вердикт
agree/disagree/unclear. Человек размечает те же тексты независимо: в форме нет ни метки
датасета, ни вердикта LLM, ни номера строки источника, порядок перемешан. Сравнение
делается после: человек против датасета (та же сводка, что у LLM) и человек против LLM.

Форма содержит тексты и поэтому пишется в reports/local/ (не публикуется). Ответы человека
(номер строки, метка, пояснение без цитат) хранятся отдельно от ответов LLM:
data/label_audit_human.csv.
"""

from __future__ import annotations

import html
import json
import math
import random
from pathlib import Path

import pandas as pd

from toxic_clf import data

AUDIT_PATH = data.PROJECT_ROOT / "data" / "label_audit.csv"
HUMAN_PATH = data.PROJECT_ROOT / "data" / "label_audit_human.csv"
FORM_PATH = data.PROJECT_ROOT / "reports" / "local" / "human_label_audit_form.html"
SHUFFLE_SEED = 2026
DEFINITION = "токсичный — оскорбление, угроза, непристойность или ненависть к группе"
HUMAN_LABELS = {"1", "0", "unclear"}


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def form_items(audit: pd.DataFrame, dataset: pd.DataFrame, seed: int = SHUFFLE_SEED) -> list[dict]:
    """Тексты выборки в перемешанном порядке; только номер строки (для выгрузки) и текст."""
    texts = dataset.set_index(data.ID_COL)[data.TEXT_COL]
    missing = sorted(set(audit[data.ID_COL]) - set(texts.index))
    if missing:
        raise ValueError(f"Строк выборки нет в датасете: {missing[:5]}")
    rows = sorted(int(r) for r in audit[data.ID_COL])
    random.Random(seed).shuffle(rows)
    return [{"row": row, "text": texts[row]} for row in rows]


def render_form(items: list[dict]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return FORM_TEMPLATE.replace("__DEFINITION__", html.escape(DEFINITION)).replace("__ITEMS__", payload)


def write_form(path: Path = FORM_PATH) -> Path:
    items = form_items(pd.read_csv(AUDIT_PATH), data.load_dataset())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_form(items), encoding="utf-8", newline="\n")
    return path


def load_human(path: Path = HUMAN_PATH) -> pd.DataFrame:
    human = pd.read_csv(path, dtype={"human_label": str, "human_note": str}, keep_default_na=False)
    bad = sorted(set(human["human_label"]) - HUMAN_LABELS)
    if bad:
        raise ValueError(f"Недопустимые метки в {path}: {bad}; ожидаются 1, 0 или unclear")
    return human


def llm_label(label: int, review: str) -> str:
    """Метка, которую подразумевает вердикт LLM о метке датасета."""
    if review == "unclear":
        return "unclear"
    return str(label if review == "agree" else 1 - label)


def compare(audit: pd.DataFrame, human: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Сводка (человек против датасета и против LLM) и таблица расхождений человека с LLM."""
    expected, got = set(audit[data.ID_COL]), set(human[data.ID_COL])
    if expected != got:
        raise ValueError(f"Строки не совпадают с выборкой: нет {sorted(expected - got)[:5]}, "
                         f"лишние {sorted(got - expected)[:5]}")
    merged = audit.merge(human, on=data.ID_COL)
    merged["llm_label"] = [llm_label(lab, rev) for lab, rev in zip(merged[data.LABEL_COL], merged["review"])]

    vs_dataset = {}
    for label, part in merged.groupby(data.LABEL_COL):
        disagree = int((part["human_label"] == str(1 - label)).sum())
        vs_dataset[f"label_{label}"] = {
            "n": int(len(part)), "disagree": disagree, "unclear": int((part["human_label"] == "unclear").sum()),
            "disagree_share": round(disagree / len(part), 3), "wilson_95": wilson_interval(disagree, len(part)),
        }

    decisive = merged[(merged["human_label"] != "unclear") & (merged["llm_label"] != "unclear")]
    h, c = decisive["human_label"].astype(int), decisive["llm_label"].astype(int)
    observed = float((h == c).mean()) if len(decisive) else float("nan")
    chance = float(h.mean() * c.mean() + (1 - h.mean()) * (1 - c.mean())) if len(decisive) else float("nan")
    kappa = (observed - chance) / (1 - chance) if len(decisive) and chance < 1 else float("nan")
    summary = {
        "method": (f"{len(merged)} текстов из data/label_audit.csv, разметка человеком вслепую: без метки датасета, "
                   "вердикта LLM и номера строки, порядок перемешан (seed 2026). Определение: " + DEFINITION),
        "human_vs_dataset": vs_dataset,
        "human_vs_llm": {
            "both_decisive": int(len(decisive)),
            "agreement": round(observed, 3),
            "cohen_kappa": round(kappa, 3),
            "confusion": {f"human_{a}_llm_{b}": int(((h == a) & (c == b)).sum()) for a in (0, 1) for b in (0, 1)},
            "unclear": {"human": int((merged["human_label"] == "unclear").sum()),
                        "llm": int((merged["llm_label"] == "unclear").sum())},
        },
    }
    discrepancies = merged[merged["human_label"] != merged["llm_label"]][
        [data.ID_COL, data.LABEL_COL, "review", "llm_label", "human_label", "note", "human_note"]
    ].rename(columns={data.LABEL_COL: "dataset_label", "review": "llm_review", "note": "llm_note"})
    return summary, discrepancies.sort_values(data.ID_COL).reset_index(drop=True)


FORM_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Слепая проверка разметки</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 16px; background: #fafafa; color: #222; }
  header { position: sticky; top: 0; background: #fafafa; padding: 8px 0; border-bottom: 1px solid #ddd; z-index: 1; }
  .item { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }
  .item.done { border-color: #8bc48b; }
  .text { white-space: pre-wrap; margin: 4px 0 10px; }
  .num { color: #888; font-size: 13px; }
  label { margin-right: 16px; cursor: pointer; }
  input[type=text] { width: 100%; box-sizing: border-box; margin-top: 6px; padding: 4px 6px; }
  button { font-size: 15px; padding: 6px 14px; }
</style>
</head>
<body>
<header>
  <b>Слепая проверка разметки.</b> Определение: __DEFINITION__.
  Отметьте, токсичен ли текст по этому определению; если по тексту нельзя решить — «неясно».
  Пояснение — по желанию и <b>без цитат</b> (файл с ответами публикуется).
  Ответы сохраняются в этом браузере автоматически.
  <div style="margin-top:6px">Отмечено: <span id="count">0</span> из <span id="total"></span>
  <button id="export">Скачать ответы (CSV)</button></div>
</header>
<main id="list"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "toxic-human-audit-v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} update(); };
const list = document.getElementById("list");
document.getElementById("total").textContent = ITEMS.length;
ITEMS.forEach((item, i) => {
  const a = answers[item.row] || {label: "", note: ""};
  const div = document.createElement("div");
  div.className = "item";
  div.innerHTML = `<div class="num">${i + 1}</div><div class="text"></div>
    <label><input type="radio" name="r${i}" value="1"> токсичный</label>
    <label><input type="radio" name="r${i}" value="0"> нетоксичный</label>
    <label><input type="radio" name="r${i}" value="unclear"> неясно</label>
    <input type="text" placeholder="пояснение без цитат (необязательно)">`;
  div.querySelector(".text").textContent = item.text;
  div.querySelectorAll("input[type=radio]").forEach(r => {
    r.checked = r.value === a.label;
    r.addEventListener("change", () => { answers[item.row] = {...(answers[item.row] || {note: ""}), label: r.value}; save(); });
  });
  const note = div.querySelector("input[type=text]");
  note.value = a.note || "";
  note.addEventListener("input", () => { answers[item.row] = {...(answers[item.row] || {label: ""}), note: note.value}; save(); });
  div.dataset.row = item.row;
  list.appendChild(div);
});
function update() {
  let n = 0;
  document.querySelectorAll(".item").forEach(d => {
    const done = !!(answers[d.dataset.row] && answers[d.dataset.row].label);
    d.classList.toggle("done", done); n += done;
  });
  document.getElementById("count").textContent = n;
}
update();
const q = s => /[",\\n\\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
document.getElementById("export").addEventListener("click", () => {
  const missing = ITEMS.filter(it => !(answers[it.row] && answers[it.row].label)).length;
  if (missing && !confirm(`Не отмечено ${missing}. Всё равно скачать?`)) return;
  const lines = ["source_row,human_label,human_note"].concat(
    ITEMS.filter(it => answers[it.row] && answers[it.row].label)
      .sort((a, b) => a.row - b.row)
      .map(it => [it.row, answers[it.row].label, q(answers[it.row].note || "")].join(",")));
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "label_audit_human.csv";
  link.click();
});
</script>
</body>
</html>
"""
