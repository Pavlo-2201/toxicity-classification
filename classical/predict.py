"""Предсказание токсичности сохранённой моделью."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from toxic_clf.final_model import MODEL_PATH, load_calibrator, load_model, toxicity_scores
from toxic_clf.cli import use_utf8_stdio


def read_texts(path: Path, column: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={column: str}, keep_default_na=False)
        if column not in df.columns:
            raise SystemExit(f"В {path} нет колонки {column!r}")
        return df
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return pd.DataFrame({column: [line for line in lines if line.strip()]})


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--text", action="append", help="текст для классификации (можно несколько раз)")
    parser.add_argument("--input", type=Path, help=".csv с колонкой текста или .txt по тексту на строку")
    parser.add_argument("--column", default="comment")
    parser.add_argument("--output", type=Path, help="куда сохранить CSV; по умолчанию stdout")
    args = parser.parse_args()

    if bool(args.text) == bool(args.input):
        parser.error("укажите либо --text, либо --input")
    df = pd.DataFrame({args.column: args.text}) if args.text else read_texts(args.input, args.column)

    model = load_model(args.model)
    texts = df[args.column].tolist()
    scores = toxicity_scores(model, texts)
    # score — сырой margin LinearSVC, не вероятность; probability — калиброванная (OOF Platt)
    df["score"] = scores.round(4)
    calibrator = load_calibrator(args.model)
    if calibrator is not None:
        df["probability"] = calibrator.predict_proba(scores).round(4)
    df["toxic"] = model.predict(texts)

    if args.output:
        df.to_csv(args.output, index=False, lineterminator="\n")
        print(f"Сохранено {len(df)} предсказаний в {args.output}")
    else:
        df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
