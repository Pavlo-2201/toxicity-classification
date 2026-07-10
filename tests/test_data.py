import pandas as pd
import pytest

from toxic_core import data


def make_source(n_other: int, rows: list[tuple[str, float]]) -> pd.DataFrame:
    other = [("другой корпус", 0.0)] * n_other
    return pd.DataFrame(other + rows, columns=[data.TEXT_COL, data.LABEL_COL])


def test_extract_subset_takes_tail_and_casts_labels(monkeypatch):
    monkeypatch.setattr(data, "SUBSET_START_ROW", 2)
    source = make_source(2, [("Текст один\n", 1.0), ("NA", 0.0)])
    subset = data.extract_subset(source)
    assert subset[data.ID_COL].tolist() == [2, 3]
    assert subset[data.LABEL_COL].tolist() == [1, 0]
    assert subset[data.LABEL_COL].dtype.kind == "i"


def test_extract_subset_rejects_unknown_labels(monkeypatch):
    monkeypatch.setattr(data, "SUBSET_START_ROW", 0)
    with pytest.raises(ValueError):
        data.extract_subset(make_source(0, [("текст", 0.5)]))


def test_read_source_keeps_na_like_texts(tmp_path):
    path = tmp_path / "src.csv"
    path.write_text("comment,toxic\nNA,0.0\nnull,1.0\n", encoding="utf-8")
    df = data.read_source_csv(path)
    assert df[data.TEXT_COL].tolist() == ["NA", "null"]


def test_check_dataset_flags_empty_and_whitespace_texts():
    df = pd.DataFrame({data.ID_COL: [0, 1, 2], data.TEXT_COL: ["ок", "", "  "], data.LABEL_COL: [0, 1, 0]})
    report = data.check_dataset(df)
    assert report["empty_texts"] == 1
    assert report["whitespace_only_texts"] == 1
    assert not report["passed"]


def test_fingerprint_ignores_row_order_but_not_content():
    df = pd.DataFrame({data.ID_COL: [0, 1], data.TEXT_COL: ["а", "б"], data.LABEL_COL: [0, 1]})
    shuffled = df.iloc[::-1]
    changed = df.assign(**{data.LABEL_COL: [1, 1]})
    assert data.dataset_fingerprint(df) == data.dataset_fingerprint(shuffled)
    assert data.dataset_fingerprint(df) != data.dataset_fingerprint(changed)


@pytest.mark.skipif(not data.DATASET_FILE.exists(), reason="датасет не скачан")
def test_downloaded_dataset_matches_fingerprint():
    df = data.load_dataset()
    assert len(df) == data.EXPECTED_ROWS
    assert data.dataset_fingerprint(df) == data.EXPECTED_FINGERPRINT
