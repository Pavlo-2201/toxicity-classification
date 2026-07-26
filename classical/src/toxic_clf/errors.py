"""Эвристические категории для анализа ошибок."""

from __future__ import annotations

import re

SHORT_MAX_CHARS = 40

# Корни русской обсценной лексики (мат) и её распространённых замен
OBSCENE = re.compile(
    r"(?<![а-яё])(х[уy][йяеёию]|пизд|п[иі]зд|[её]б[аеиуы]|[её]бл|\w*[её]б(ан|ну|ат|ыв|ал)|бля|бляд|муд[аио]|залуп|"
    r"пид[оа]р|сук[аи]?(?![а-яё])|сучк)",
    re.IGNORECASE,
)
# Явные оскорбления без мата
INSULT = re.compile(
    r"(идиот|дебил|дур[ао]к|дура(?![а-яё])|урод|туп[оыа]|мраз|твар|скот|быдл|чмо|лох|петух|долба|даун|"
    r"дегенерат|кретин|ублюд|гнид|выродок|шлюх|ничтожеств|хохл|хач|жид|чурк|пидр|педик|свин)",
    re.IGNORECASE,
)
# Признаки намеренного искажения: маскировка точками/звёздочками, латиница внутри кириллицы, растянутые буквы
OBFUSCATION = re.compile(
    r"([а-яё][*.]{1,3}[а-яё])|([а-яё][a-z]|[a-z][а-яё])|(([а-яё])\4{2,})",
    re.IGNORECASE,
)


def categorize(text: str, oov_share: float) -> dict[str, bool]:
    stripped = text.strip()
    return {
        "short": len(stripped) <= SHORT_MAX_CHARS,
        "obscene": bool(OBSCENE.search(stripped)),
        "insult_lexicon": bool(INSULT.search(stripped)),
        "obfuscation_or_typos": bool(OBFUSCATION.search(stripped)) or oov_share >= 0.3,
        "no_explicit_markers": not OBSCENE.search(stripped) and not INSULT.search(stripped),
    }
