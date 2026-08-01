import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
TEXT = "ёжик 😀 naïve — «кавычки»"
CODE = f"from toxic_clf.cli import use_utf8_stdio; use_utf8_stdio(); print(input()); print({TEXT!r})"
# кодировка локали как у Windows с русской раскладкой: без исправления печать эмодзи падает
ENV = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "cp1251", "PYTHONUTF8": "0"}


def test_stdio_is_utf8_in_pipe():
    out = subprocess.run([sys.executable, "-c", CODE], input=("\ufeff" + TEXT + "\n").encode("utf-8"),
                         capture_output=True, env=ENV, check=True)
    assert out.stdout.decode("utf-8").splitlines() == [TEXT, TEXT]


def test_stdio_is_utf8_in_file(tmp_path):
    path = tmp_path / "out.txt"
    with path.open("wb") as stream:
        subprocess.run([sys.executable, "-c", CODE], input=(TEXT + "\n").encode("utf-8"),
                       stdout=stream, env=ENV, check=True)
    assert path.read_text(encoding="utf-8").splitlines() == [TEXT, TEXT]
