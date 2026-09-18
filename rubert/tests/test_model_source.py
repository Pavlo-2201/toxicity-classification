import json
import shutil

import pytest

from rubert_clf import data, model_source


def test_registered_tiny_is_verified_before_loading(tokenizer):
    pinned = model_source.verify_registered(model_source.MODEL_DIR)
    assert pinned is model_source.DEFAULT and pinned.pinned.endswith(model_source.REVISION)


def test_unregistered_dir_is_not_checked(tmp_path):
    assert model_source.registered(tmp_path) is None
    assert model_source.verify_registered(tmp_path) is None


def test_changed_file_in_registered_dir_is_rejected(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr(model_source, "MODELS_ROOT", tmp_path)
    target = tmp_path / "rubert-tiny2"
    target.mkdir()
    for name in model_source.FILES:
        if name != "model.safetensors":
            shutil.copyfile(data.PROJECT_ROOT / "models" / "rubert-tiny2" / name, target / name)
    (target / "model.safetensors").write_bytes(b"not the pinned weights")
    with pytest.raises(ValueError, match="model.safetensors"):
        model_source.verify_registered(target)


def test_missing_registered_model_points_to_downloader(tmp_path, monkeypatch):
    monkeypatch.setattr(model_source, "MODELS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="--model ruBert-base"):
        model_source.verify_registered(tmp_path / "ruBert-base")


def test_protocol_models_are_all_in_registry():
    protocol = json.loads((data.PROJECT_ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
    for candidate in protocol["candidates"]:
        assert model_source.registered(data.PROJECT_ROOT / candidate["model_dir"]) is not None, candidate["name"]
    for ready in protocol["ready_classifiers"]:
        pinned = model_source.registered(data.PROJECT_ROOT / ready["dir"])
        assert pinned is not None and (pinned.model_id, pinned.revision) == (ready["name"], ready["revision"])
    final = json.loads((data.PROJECT_ROOT / "configs" / "final_base.json").read_text(encoding="utf-8"))
    assert final["candidate"] in {c["name"] for c in protocol["candidates"]}
