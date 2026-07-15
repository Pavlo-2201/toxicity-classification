import pandas as pd
import pytest

from toxic_core import data, results, splits


def toy_dataset(n: int = 200) -> pd.DataFrame:
    texts = [f"Комментарий номер {i}" for i in range(n)]
    labels = [int(i % 3 == 0) for i in range(n)]
    df = pd.DataFrame({data.ID_COL: range(n), data.TEXT_COL: texts, data.LABEL_COL: labels})
    extra = pd.DataFrame(
        {
            data.ID_COL: [n, n + 1, n + 2],
            # копия с хвостовым переводом строки и другим регистром; копия с противоречивой меткой
            data.TEXT_COL: ["КОММЕНТАРИЙ номер 1\n", "Комментарий  номер 2", "Комментарий номер 3 "],
            data.LABEL_COL: [0, 0, 0],
        }
    )
    return pd.concat([df, extra], ignore_index=True)


def test_duplicate_key_ignores_whitespace_and_case():
    assert splits.duplicate_key("Привет,  Мир!\n") == splits.duplicate_key("привет, мир!")
    assert splits.duplicate_key("привет, мир") != splits.duplicate_key("привет мир")


def test_resolve_duplicates_rules():
    resolved = splits.resolve_duplicates(toy_dataset()).set_index(data.ID_COL)["exclusion_reason"]
    assert resolved[200] == "duplicate_of_1"  # совпадает с 1, метки одинаковые
    assert resolved[1] == ""  # первый экземпляр сохраняется
    assert resolved[201] == "duplicate_of_2"
    assert resolved[3] == resolved[202] == "conflicting_labels"  # 3 токсичный, копия нет


def test_splits_disjoint_cover_and_stratified():
    manifest = splits.build_manifest(toy_dataset())
    kept = manifest[manifest["split"] != splits.EXCLUDED]
    assert set(kept["split"]) == set(splits.SPLITS)
    assert kept["group_sha256"].is_unique  # одинаковые тексты не попадают в разные разбиения
    assert len(manifest) == len(toy_dataset())
    shares = kept.groupby("split")[data.LABEL_COL].mean()
    assert (shares - kept[data.LABEL_COL].mean()).abs().max() < 0.05


def test_manifest_is_deterministic():
    df = toy_dataset()
    pd.testing.assert_frame_equal(splits.build_manifest(df), splits.build_manifest(df.sample(frac=1, random_state=1)))


def test_load_splits_detects_changed_text():
    df = toy_dataset()
    manifest = splits.build_manifest(df)
    tampered = df.copy()
    tampered.loc[5, data.TEXT_COL] = "другой текст"
    with pytest.raises(ValueError):
        splits.load_splits(tampered, manifest)


needs_data = pytest.mark.skipif(not data.DATASET_FILE.exists(), reason="данные не подготовлены: toxic prepare")


@needs_data
def test_committed_manifest_matches_data():
    manifest = splits.load_manifest()
    assert manifest.equals(splits.build_manifest(data.load_dataset()))
    parts = splits.load_splits(manifest=manifest)
    keys = {name: set(part[data.TEXT_COL].map(splits.duplicate_key)) for name, part in parts.items()}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]


# --- закреплённый манифест (общий для classical и rubert) ---


def test_manifest_sha_is_pinned():
    manifest = splits.load_manifest()  # бросит исключение при несовпадении SHA-256
    assert len(manifest) == data.EXPECTED_ROWS


def test_no_text_shared_between_splits():
    overlaps = splits.split_overlaps(splits.load_manifest())
    assert all(v == 0 for v in overlaps.values()), overlaps


def test_split_sizes_match_fingerprint():
    manifest = splits.load_manifest()
    counts = splits.load_fingerprint()["counts"]
    for name in (*splits.SPLITS, splits.EXCLUDED):
        assert (manifest.split == name).sum() == counts[name]["rows"]


@needs_data
def test_rebuilt_manifest_identical():
    report = splits.compare_manifests(splits.load_manifest(), splits.build_manifest(data.load_dataset()))
    assert report["identical"], report


@needs_data
def test_load_splits_detects_changed_dataset_row():
    changed = data.load_dataset()
    changed.loc[0, data.TEXT_COL] = changed.loc[0, data.TEXT_COL] + "!"
    with pytest.raises(ValueError):
        splits.load_splits(changed)


def test_compare_manifests_detects_moved_row():
    manifest = splits.load_manifest()
    moved = manifest.copy()
    idx = moved.index[moved.split == "test"][0]
    moved.loc[idx, "split"] = "train"
    report = splits.compare_manifests(manifest, moved)
    assert not report["identical"] and report["mismatches_by_column"]["split"] == 1


def test_overlap_detection():
    toy = pd.DataFrame(
        {"split": ["train", "test", "validation"], "text_sha256": ["a", "a", "b"], "group_sha256": ["g1", "g2", "g3"]}
    )
    assert splits.split_overlaps(toy)["train-test:text_sha256"] == 1


@needs_data
def test_split_files_are_rebuilt_byte_for_byte():
    """`toxic prepare --rebuild-splits` дал бы те же байты: манифест и data_fingerprint.json — производные данных."""
    df = data.load_dataset()
    assert results.csv_text(splits.build_manifest(df)).encode("utf-8") == splits.MANIFEST_PATH.read_bytes()
    assert results.json_text(splits.fingerprint_info(df)).encode("utf-8") == splits.FINGERPRINT_PATH.read_bytes()


def test_fingerprint_file_pins_the_same_manifest():
    info = splits.load_fingerprint()
    assert info["manifest_sha256"] == splits.MANIFEST_SHA256 == data.sha256_file(splits.MANIFEST_PATH)
    assert info["content_fingerprint"] == data.EXPECTED_FINGERPRINT
    assert info["manifest_file"] == data.relative_to(splits.MANIFEST_PATH) == "data/splits/split_manifest.csv"
