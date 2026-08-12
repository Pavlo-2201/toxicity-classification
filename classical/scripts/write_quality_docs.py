"""Генерирует docs/data_quality.md и docs/external_and_calibration.md из отчётов в reports/.

Числа в документах берутся только из JSON/CSV отчётов, чтобы текст не расходился с ними.
"""

from __future__ import annotations

import json

import pandas as pd

from toxic_clf import data
from toxic_clf.cli import use_utf8_stdio

ROOT = data.PROJECT_ROOT
REPORTS = ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def data_quality() -> str:
    c = pd.read_csv(REPORTS / "duplicate_conflicts.csv")
    cnt = c.review_category.value_counts().to_dict()
    a = load("label_audit.json")
    gaps = [abs(int(x) - int(y)) for x, y in (r.split() for r in c.source_rows)]
    rows = []
    for y in (0, 1):
        s = a[f"label_{y}"]
        rows.append(f"| {y} | {s['n']} | {s['disagree']} | {s['unclear']} | {pct(s['disagree_share'])} "
                    f"[{pct(s['wilson_95'][0])}; {pct(s['wilson_95'][1])}] |")
    return f"""# Качество разметки

Две независимые проверки. Обе дают сигнал о шуме в своих группах текстов, а не оценку доли
ошибок разметки во всём датасете.

## 43 конфликтующие пары дубликатов

После дедупликации (NFC, пробелы, casefold) {len(c)} групп текстов имеют разные метки, и каждая
группа — ровно две строки. Они исключены из train/validation/test целиком ({2 * len(c)} строк).
Разбор — `reports/duplicate_conflicts.csv`: номера строк, какая копия помечена токсичной, длина
текста и категория. Самих текстов в файле нет.

Категории (тексты прочитаны моделью LLM):

| категория | групп | что это |
|---|---|---|
| `neutral_text` | {cnt['neutral_text']} | нейтральный текст (новость, вопрос, бытовая реплика), одна копия помечена токсичной |
| `borderline_rudeness` | {cnt['borderline_rudeness']} | резкость без мата и без явного оскорбления («шлак полный» о фильме) |
| `obscenity_not_addressed` | {cnt['obscenity_not_addressed']} | мат не в адрес собеседника |
| `explicit_insult_or_hate` | {cnt['explicit_insult_or_hate']} | явное оскорбление или ненависть к группе, одна копия помечена нетоксичной |

Наблюдения:

- в половине групп текст по содержанию нейтральный: шум здесь в основном ложные «1»;
- резкость и мат «в воздух» — граница определения токсичности, а не очевидные ошибки:
  по-видимому, её проводили по-разному;
- копии лежат далеко друг от друга (медианная разница номеров строк {int(pd.Series(gaps).median())}),
  номера образуют блоки 150xxx против 155xxx/157xxx/163xxx — похоже на повторно размеченные партии.

Это 0,6 % строк, причём особые: дословные повторы, размеченные дважды. Переносить их долю на весь
датасет нельзя.

## Ручная проверка выборки разметки

120 текстов train, по 60 каждой метки (seed 2026), без конфликтных дубликатов. Определение:
токсичный — оскорбление, угроза, непристойность или ненависть к группе. Результат —
`data/label_audit.csv` (номер строки, метка, вердикт, пояснение без цитат), сводка —
`reports/label_audit.json`.

| метка в датасете | текстов | не согласен | неясно | доля несогласия, 95 % ДИ (Уилсон) |
|---|---|---|---|---|
{chr(10).join(rows)}

Тексты с меткой 1, с которыми проверка не согласна: лозунги и призывы без оскорблений, вежливые
вопросы, рассуждения о группах людей без оскорбительных слов. Это согласуется с разбором
конфликтов: шум сосредоточен в положительном классе. Часть «пропусков» модели на test — тексты
без признаков токсичности.

**Кто проверял.** Проверку выполнила модель LLM по определению выше, человеком она не
перепроверялась. Выборка мала, интервалы широкие: читать стоит интервал, а не точку.
"""


def external_and_calibration() -> str:
    ext, fm = load("external_okru.json"), load("final_metrics.json")
    m, ov, t = ext["metrics_at_model_threshold"], ext["overlap_check"], fm["test"]
    cal, raw = t["calibration_oof_platt"], t["calibration_raw_sigmoid"]
    return f"""# Внешняя оценка (ok.ru) и калибровка

## Калибровка вероятностей

LinearSVC выдаёт margin — расстояние до разделяющей гиперплоскости. В отчётах он остаётся
столбцом `score` и вероятностью не является. Вероятность — столбец `probability`:
Platt-шкалирование (логистическая функция от margin) с параметрами, подобранными на
out-of-fold margin обучающей выборки (5-fold). Validation и test в калибровке не участвуют.
Параметры хранятся в `models/toxic_clf.json`, `predict.py` выводит оба столбца. Решения модели не
меняются: порог задан на margin.

| test | Brier | log-loss | ECE (10 интервалов) |
|---|---|---|---|
| OOF Platt | {cal['brier']} | {cal['log_loss']} | {cal['ece_10_bins']} |
| логистическая функция от сырого margin | {raw['brier']} | {raw['log_loss']} | {raw['ece_10_bins']} |

## ok.ru

Набор: Toxic Russian Comments (комментарии ok.ru, соревнование OK ML Cup), автор Alexander
Semiletov, Kaggle `alexandersemiletov/toxic-russian-comments`, лицензия CC BY-NC-SA 4.0.
Используется test-часть зеркала на Hugging Face `{ext['dataset']['mirror_repo']}` (ревизия
`{ext['dataset']['revision'][:12]}`, SHA-256 закреплён в `src/toxic_clf/external.py`). Тексты в нижнем
регистре; метка 1 — оскорбление, угроза или непристойность. Тексты в репозиторий не попадают.

Проверка пересечений (ключ: нижний регистр, ё→е, только буквы и цифры):

| строк | совпали с нашим корпусом | дубликаты внутри ok.ru | осталось | доля токсичных |
|---|---|---|---|---|
| {ov['rows']} | {ov['overlap_with_our_corpus']} | {ov['duplicate_rows_dropped']} | {ov['rows_kept']} | {pct(ov['toxic_share_kept'])} |

Итоговая модель обучена только на train нашего корпуса; порог и калибровка не менялись:

| | ROC-AUC | AP | F1 toxic | precision | recall | доля ответов «токсичный» |
|---|---|---|---|---|---|---|
| test нашего корпуса | {t['roc_auc']} | {t['average_precision']} | {t['f1_toxic']} | {t['precision_toxic']} | {t['recall_toxic']} | {pct(t['predicted_toxic_share'])} |
| ok.ru | {ext['roc_auc']} | {ext['average_precision']} | {m['f1_toxic']} [{m['f1_toxic_bootstrap_ci']['low']}; {m['f1_toxic_bootstrap_ci']['high']}] | {m['precision_toxic']} | {m['recall_toxic']} | {pct(m['predicted_toxic_share'])} |

Ранжирование переносится (ROC-AUC {ext['roc_auc']}), порог и вероятности — нет: модель называет
токсичными {pct(m['predicted_toxic_share'])} текстов при {pct(ov['toxic_share_kept'])} в разметке, ECE
вероятностей {m['calibration_oof_platt']['ece_10_bins']}. Порог, подобранный на самих данных ok.ru, дал бы
F1 {ext['oracle_threshold_on_okru']['f1_toxic']}; это верхняя граница, а не результат. Причины,
видимые при локальном просмотре ложных срабатываний:

- **другое определение и разметка**: среди самых уверенных ложных срабатываний — явные
  оскорбления и этнические ярлыки, помеченные в ok.ru как нейтральные;
- **другой домен**: короткие бытовые реплики с эмодзи и без пунктуации получают margin около
  порога, а в 2ch/pikabu таких текстов мало;
- **другая доля токсичных**: порог и калибровка подобраны под 33 % токсичных, а здесь 18 %.

Для готовых нейросетевых классификаторов такая оценка нечестна:
`cointegrated/rubert-tiny-toxicity` и `s-nlp/russian_toxicity_classifier` обучались в том числе на
ok.ru (см. `rubert-text-classification`).

Попримерные результаты без текстов: `reports/external_okru_predictions.csv`, сводка —
`reports/external_okru.json`.
"""


def main() -> None:
    use_utf8_stdio()
    (ROOT / "docs" / "data_quality.md").write_text(data_quality(), encoding="utf-8", newline="\n")
    (ROOT / "docs" / "external_and_calibration.md").write_text(external_and_calibration(), encoding="utf-8", newline="\n")
    print("ok")


if __name__ == "__main__":
    main()
