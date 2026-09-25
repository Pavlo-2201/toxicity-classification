# russian-toxicity-classification

Бинарная классификация токсичных русскоязычных комментариев (2ch.hk, pikabu.ru) и воспроизводимое
сравнение классической модели (TF-IDF + LinearSVC) с дообученным RuBERT. У обеих моделей одни данные,
один манифест разбиений, одни метрики и один формат оценки.

## Сравнение моделей

Итоговые артефакты обеих частей на одном test, общий формат (`toxic evaluate`, `toxic compare`):

| модель | порог выбран | F1 toxic | precision | recall | AP | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF char 2–5 + LinearSVC | по OOF-оценкам train | 0.854 [0.834; 0.873] | 0.833 | 0.877 | 0.927 | 0.961 |
| **ruBert-base, L=128** (seed 42, эпоха 2) | на калибровочной части train | **0.891** [0.874; 0.907] | 0.893 | 0.888 | 0.954 | 0.976 |

- В квадратных скобках — 95 % бутстреп-интервал (1000 повторов, seed 42).
- Разность F1 (RuBERT − TF-IDF): +0.036 [+0.017; +0.057], парный бутстреп. Тест Макнемара:
  132 текста верно классифицирует только RuBERT, 75 — только TF-IDF, p = 0.0001
  (`reports/eval/compare-classical-vs-rubert_test.json`).
- Устойчивость результата к seed (протокол, 3 seed на конфигурацию):

| конфигурация | test F1 | test AP | CPU, мс/текст | обучаемых параметров |
|---|---|---|---|---|
| rubert-tiny2, L=256 | 0.886 ± 0.006 | 0.947 ± 0.003 | 2.31 | 29.2 M |
| ruBert-base + LoRA, L=128 | 0.885 ± 0.013 | 0.953 ± 0.001 | 48.84 | 0.6 M |
| ruBert-base, L=128 | 0.891 ± 0.002 | 0.954 ± 0.002 | 33.43 | 178.3 M |

ruBert-base лучше TF-IDF на 0.038 F1 в среднем по трём seed, тест Макнемара p ≤ 0.0002 при каждом
seed. На CPU он примерно в 14 раз дороже rubert-tiny2 с L=256, а по качеству разница невелика.

### Протоколы выбора

Протоколы у моделей разные, и это сохранено намеренно. Порог хранится в артефакте, а `toxic evaluate`
его только применяет и записывает в отчёт, как он выбран.

| | classical | rubert (протокол) |
|---|---|---|
| модель / конфигурация | 5-fold CV на train | AP на validation, среднее по 3 seed |
| эпоха | — | максимум AP на validation (от порога не зависит) |
| порог | F1-оптимум по OOF-оценкам train (5-fold); принимается, если F1 на validation не ниже, чем при пороге 0 | F1-оптимум на калибровочной части train (15 %, стратифицированно, seed 42; модель на ней не обучается) |
| score | margin LinearSVC; вероятность — Platt на OOF train | P(toxic), softmax |
| test | только итоговая оценка | только итоговая оценка |

Тесты проверяют это и на деле. Если перевернуть метки test, порог, калибровка и модель classical не
меняются (`classical/tests/test_train_protocol.py`), как и эпоха, порог и предсказания validation у
RuBERT (`rubert/tests/test_protocol.py`).

## Архитектура

```
src/toxic_core/            общее ядро (без PyTorch)
  data.py                  источник: закреплённый коммит s-nlp/rudetoxifier, SHA-256, отпечаток содержимого
  splits.py                манифест разбиений: построение, SHA-256, проверка хешей текстов, отсутствие пересечений
  metrics.py               метрики, бутстреп, выбор порога по F1, Макнемар, парный бутстреп
  text.py, stdio.py        нормализация текста, UTF-8 в консоли Windows
  results.py               общий формат оценки и сравнения, защита от перезаписи
  registry.py, cli.py      единая команда `toxic`, ленивая загрузка моделей
classical/                 TF-IDF + линейные модели: src/toxic_clf (модели, выбор, калибровка, устойчивость),
                           train.py, predict.py, scripts/ экспериментов, reports/, docs/
rubert/                    RuBERT: src/rubert_clf (Dataset, обучение, протокол, LoRA, реестр моделей),
                           train.py, predict.py, evaluate.py, configs/, scripts/, reports/, docs/
data/splits/               единственный манифест разбиений и data_fingerprint.json (происхождение данных)
reports/data/              проверки данных и разбиений (`toxic prepare`)
reports/eval/              оценки обеих моделей и их сравнения в общем формате
tests/                     ядро, общий CLI, согласованность данных
```

Ядро отвечает за загрузку и проверку данных, разбиения, метрики, нормализацию текста и формат результатов.
Каждая модель содержит свои процедуры обучения. У classical это конфигурации TF-IDF,
CV, OOF-порог, калибровка, варианты нормализации и внешний набор. У rubert — токенизация, цикл
обучения, протокол, каталоги запусков, реестр моделей и сверка артефакта с отчётом прогона. Каждая
часть подключается к ядру через адаптер (`toxic_clf/adapter.py`, `rubert_clf/adapter.py`), где
описаны загрузка артефакта, `score`, `probability` и протокол выбора.

## Установка

Из корня репозитория, Python 3.11+; проверенное окружение и CI — Python 3.13.12. Пути к данным, моделям и отчётам считаются
от исходников, поэтому пакет ставится только в editable-режиме.

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\Activate.ps1

# только классическая модель и ядро — без PyTorch
pip install -r requirements/classical.txt && pip install --no-deps -e .

# с RuBERT: сначала torch под своё железо (здесь CPU), затем остальное
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/rubert.txt && pip install --no-deps -e .
```

`requirements/*.txt` закрепляют версии для воспроизведения.
Без закрепления можно поставить
`pip install -e .` или `pip install -e ".[rubert]"`. Без зависимостей RuBERT команды для `rubert`
сообщают, что нужно доустановить, а его тесты не собираются.

## Команды

Команда `toxic` (то же самое — `python -m toxic_core`), модель задаётся позиционно:

```bash
toxic prepare                         # данные: SHA-256 исходника, отпечаток, сверка с манифестом -> reports/data/
toxic prepare rubert                  # + энкодер rubert-tiny2 по ревизии (--all-models — все модели протокола)
toxic train classical                 # аргументы после модели идут в classical/train.py (--normalization all, --output, ...)
toxic train rubert --smoke            # аргументы после модели идут в rubert/train.py (--config, --device, ...)
toxic predict classical --text "Заткнись уже" --text "Спасибо за статью"   # CSV: score, probability, toxic
toxic predict rubert --input comments.csv --column comment --output pred.csv
toxic evaluate classical --split test                 # reports/eval/classical_test.json + _predictions.csv
toxic evaluate rubert --split test --device cpu       # reports/eval/rubert_test.json + _predictions.csv
toxic compare classical rubert --split test           # парное сравнение на тех же текстах
toxic check                           # обе части читают один манифест, отчёты и артефакты ссылаются на него
pytest
```

Формат оценки одинаков у обеих моделей. `<имя>_<разбиение>_predictions.csv` содержит `source_row`,
`toxic`, `score`, `probability` и `pred` (`pred = score ≥ threshold`). В `<имя>_<разбиение>.json` —
артефакт, SHA-256 манифеста, порог и протокол его выбора, метрики, бутстреп-интервал и SHA-256 файла
предсказаний. Поле `environment` содержит версии Python и библиотек фактического расчёта.
Перед оценкой проверяется, что артефакт обучен на манифесте проекта. `compare`
отказывается сравнивать оценки на разных текстах, метках или данных.

Готовые результаты не перезаписываются случайно. `train` останавливается до начала обучения, если
артефакт или отчёты уже есть. `evaluate`, `compare` и `prepare` оставляют совпадающий по байтам файл
как есть, а отличающийся заменяют только с `--overwrite`. Дополнительные эксперименты описаны ниже и в документации моделей.

## Данные

Russian Language Toxic Comments (Kaggle `blackmoon/russian-language-toxic-comments`, GPL-2.0) — файл
из [s-nlp/rudetoxifier](https://github.com/s-nlp/rudetoxifier) на закреплённом коммите
`aef59899`. При загрузке проверяются SHA-256 файла, состав блока (14 412 строк) и отпечаток содержимого.
Дубликаты (после NFC, схлопывания пробелов и casefold) сводятся к одному тексту (214 копий
исключены), группы с противоречивыми метками исключаются целиком (86 строк). Оставшиеся тексты стратифицированно делятся
70/15/15 (seed 42): train 9 878, validation 2 117, test 2 117. Манифест
`data/splits/split_manifest.csv` хранит только хеши, метки и разбиения. Его SHA-256 закреплён в коде
и проверяется при каждом чтении, хеши текстов сверяются построчно, а из данных он пересобирается
побайтно. Тексты комментариев в репозиторий не попадают: публикуемые CSV содержат только
идентификаторы, метки и предсказания.

## Тесты и CI

`pytest` из корня собирает `tests/` (ядро, общий формат, сквозной CLI, согласованность данных),
`classical/tests` и `rubert/tests` (последние — только если установлен torch). Workflow
[`ci`](.github/workflows/ci.yml) проверяет Ubuntu и Windows двумя заданиями:

- `classical` — установка без PyTorch, `toxic prepare`, тесты, обучение и предсказание через pipe.
  Ещё в нём проверяются: побайтное воспроизведение отчётов classical и `reports/eval`, отказ
  перезаписать артефакт, `toxic check`.
- `rubert` — CPU-torch, все тесты, включая smoke `toxic train rubert --smoke` → `predict` →
  `evaluate`, и `toxic check`.

Тяжёлые эксперименты запускаются в ручном workflow
[`rubert-experiments`](.github/workflows/rubert-experiments.yml) на GPU.

## Эксперименты и документация

Команды запускаются из корня репозитория. Для варианта с нормализацией используются свой
артефакт и собственные порог и калибровка:

```bash
toxic train classical --normalization all
toxic predict classical --artifact classical/models/toxic_clf_all.joblib --text "ты тупoй идиoт"
python classical/scripts/external_eval.py --model classical/models/toxic_clf_all.joblib
```

Скрипты экспериментов рассчитаны на запуск из соответствующего каталога:

```bash
cd classical
python scripts/robustness_eval.py --evaluate-test
python scripts/external_eval.py
python scripts/error_analysis.py
python scripts/human_audit.py form
python scripts/human_audit.py compare
cd ../rubert
python scripts/experiments.py --out experiments/runs/my-run
python scripts/summarize_experiments.py --run-dir experiments/runs/my-run
```
