# russian-toxicity-classification

Классификация токсичных русскоязычных комментариев на общих данных и разбиениях.

`src/toxic_core/` содержит источник данных, разбиения, метрики, нормализацию текста и сведения об окружении.
`classical/` содержит классические модели, `rubert/` — нейросетевые эксперименты.
Каждая модель использует данные и служебные функции общего ядра.

Окружение закреплено в `.python-version` и `requirements/`. Установка из корня:

```bash
pip install -r requirements/classical.txt
pip install --no-deps -e .
python classical/scripts/download_data.py
```

Манифест разбиений и отпечаток данных хранятся в `data/splits/`, без исходных текстов.
