"""Char-level TF-IDF, word/char/word+char, class_weight и влияние дисбаланса."""

from toxic_clf.experiments import run_stage

MODELS = {"logreg": {"C": 16.0}, "linearsvc": {"C": 1.0}}
FEATURES = {
    "word12": {"word": {"ngram_range": (1, 2)}},
    "charwb25": {"char": {"analyzer": "char_wb", "ngram_range": (2, 5)}},
    "char15": {"char": {"analyzer": "char", "ngram_range": (1, 5)}},
    "word12+charwb25": {"word": {"ngram_range": (1, 2)}, "char": {"analyzer": "char_wb", "ngram_range": (2, 5)}},
}

CONFIGS = []
for feat_name, feat in FEATURES.items():
    for model, params in MODELS.items():
        for class_weight in (None, "balanced"):
            suffix = "_balanced" if class_weight else ""
            CONFIGS.append(
                {
                    "name": f"{feat_name}_{model}{suffix}",
                    "model": model,
                    "model_params": {**params, "class_weight": class_weight},
                    **feat,
                }
            )

# Проверка эффекта регистра и пунктуации на символьных признаках (LinearSVC, balanced)
ABLATION_BASE = {"model": "linearsvc", "model_params": {"C": 1.0, "class_weight": "balanced"}}
CONFIGS += [
    {"name": "charwb25_linearsvc_balanced_cased", **ABLATION_BASE,
     "char": {"analyzer": "char_wb", "ngram_range": (2, 5), "lowercase": False}},
    {"name": "charwb25_linearsvc_balanced_no_punct", **ABLATION_BASE, "strip_punctuation": True,
     "char": {"analyzer": "char_wb", "ngram_range": (2, 5)}},
    {"name": "word12_linearsvc_balanced_punct_tokens", **ABLATION_BASE,
     "word": {"ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b|[^\w\s]+"}},
]

if __name__ == "__main__":
    run_stage("stage5_char_models", CONFIGS)
