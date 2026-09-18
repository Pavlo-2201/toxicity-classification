"""Проверка каталога запуска scripts/experiments.py.

    python scripts/check_experiment_run.py experiments/runs/<id>
    python scripts/check_experiment_run.py experiments/runs/<id> --trained-in-last-session   # ручной workflow

Все запланированные прогоны выполнены этим запуском, отчёты согласованы с manifest.json, у
кандидатов пройдены все эпохи протокола. С --trained-in-last-session все прогоны должны быть
выполнены в последней сессии: обучение действительно прошло в этом задании.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rubert_clf import experiment_run as er
from rubert_clf.cli import use_utf8_stdio


def main() -> None:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trained-in-last-session", action="store_true")
    args = parser.parse_args()

    problems = er.verify(args.run_dir, args.trained_in_last_session)
    if problems:
        raise SystemExit("Запуск не прошёл проверку:\n  " + "\n  ".join(problems))
    manifest = er.read_json(args.run_dir / er.MANIFEST)
    for i, session in enumerate(manifest["sessions"], 1):
        env = session["environment"]
        print(f"сессия {i} ({session['mode']}, {session['started']}, {env['device']} {env['gpu'] or ''}): "
              f"{len(session['completed'])} прогонов")
    for change in er.environment_changes(manifest):
        print(f"окружение менялось: {change}")
    print(f"OK: {len(manifest['planned'])} прогонов, run_id {manifest['run_id']}")


if __name__ == "__main__":
    main()
