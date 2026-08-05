"""Варианты нормализации и контролируемые искажения текста для проверки устойчивости.

Нормализация baseline (`text.normalize_text`) не меняется: варианты ниже включаются
явно и сравниваются с ней. Каждый вариант трогает только узкий класс случаев, чтобы
его потери на обычных текстах можно было измерить отдельно.

Искажения имитируют то, как пишут в комментариях, и не меняют смысл: латиница-двойник
внутри кириллического слова, маскировка буквы, растянутые буквы, опечатки. Генерация
детерминирована (seed), поэтому набор устойчивости воспроизводится из скачанного
источника и не хранится в репозитории.
"""

from __future__ import annotations

import random
import re

from toxic_clf.text import normalize_text

# Латинские буквы, которые выглядят как кириллические (в нижнем и верхнем регистре)
LATIN_TO_CYRILLIC = str.maketrans("aceopxykmtbhACEOPXYKMTBH", "асеорхукмтвнАСЕОРХУКМТВН")
CYRILLIC_TO_LATIN = dict(zip("асеорхук", "aceopxyk"))

WORD = re.compile(r"\w+")
HAS_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
HAS_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
# Маска между кириллическими буквами: «д*бил», «х.у.й», «с_ука»
MASK_INSIDE_WORD = re.compile(r"(?<=[а-яё])[*._]+(?=[а-яё])", re.IGNORECASE)
STRETCHED = re.compile(r"([а-яёa-z])\1{2,}", re.IGNORECASE)


def map_homoglyphs(text: str) -> str:
    """Латиница-двойник -> кириллица, только в словах, где уже есть кириллица."""
    def fix(m: re.Match) -> str:
        word = m.group(0)
        if HAS_CYRILLIC.search(word) and HAS_LATIN.search(word):
            return word.translate(LATIN_TO_CYRILLIC)
        return word
    return WORD.sub(fix, text)


def remove_inner_masks(text: str) -> str:
    """Убирает *, точки и подчёркивания между кириллическими буквами. Задевает и «т.е.»."""
    return MASK_INSIDE_WORD.sub("", text)


def collapse_repeats(text: str) -> str:
    """Три и более одинаковых буквы подряд -> одна («дураааак» -> «дурак»)."""
    return STRETCHED.sub(r"\1", text)


NORMALIZATIONS = {
    "baseline": lambda t: normalize_text(t),
    "homoglyph": lambda t: map_homoglyphs(normalize_text(t)),
    "unmask": lambda t: remove_inner_masks(normalize_text(t)),
    "collapse": lambda t: collapse_repeats(normalize_text(t)),
    "all": lambda t: collapse_repeats(remove_inner_masks(map_homoglyphs(normalize_text(t)))),
}


class Normalizer:
    """Нормализация списка текстов по имени варианта; сериализуется joblib (в отличие от lambda)."""

    def __init__(self, name: str):
        if name not in NORMALIZATIONS:
            raise ValueError(f"Неизвестная нормализация: {name}; есть {sorted(NORMALIZATIONS)}")
        self.name = name

    def __call__(self, texts) -> list[str]:
        fn = NORMALIZATIONS[self.name]
        return [fn(t) for t in texts]


# --- Искажения --------------------------------------------------------------------

VOWELS = set("аеёиоуыэюяaeiouy")


def _eligible(word: str, min_len: int) -> bool:
    return len(word) >= min_len and bool(HAS_CYRILLIC.search(word)) and word.isalpha()


def _homoglyph_word(word: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(word) if ch.lower() in CYRILLIC_TO_LATIN]
    if not positions:
        return word
    i = rng.choice(positions)
    latin = CYRILLIC_TO_LATIN[word[i].lower()]
    return word[:i] + (latin.upper() if word[i].isupper() else latin) + word[i + 1:]


def _mask_word(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + rng.choice("*.") + word[i + 1:]


def _repeat_word(word: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(word) if ch.lower() in VOWELS] or list(range(len(word)))
    i = rng.choice(positions)
    return word[:i + 1] + word[i] * rng.randint(2, 4) + word[i + 1:]


def _typo_word(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 2)
    if rng.random() < 0.5:  # перестановка соседних букв
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i + 1:]  # пропуск буквы


PERTURBATIONS = {
    # имя: (функция над словом, минимальная длина слова, доля затронутых слов)
    "homoglyph": (_homoglyph_word, 3, 0.3),
    "mask": (_mask_word, 4, 0.2),
    "repeat": (_repeat_word, 3, 0.2),
    "typo": (_typo_word, 5, 0.15),
}


def perturb(text: str, kind: str, seed: int) -> str:
    """Искажение одного текста. kind — ключ PERTURBATIONS или "combo" (все подряд)."""
    if kind == "combo":
        for i, name in enumerate(PERTURBATIONS):
            text = perturb(text, name, seed * 31 + i)
        return text
    fn, min_len, share = PERTURBATIONS[kind]
    rng = random.Random(seed)

    def edit(m: re.Match) -> str:
        word = m.group(0)
        if _eligible(word, min_len) and rng.random() < share:
            return fn(word, rng)
        return word
    return WORD.sub(edit, text)
