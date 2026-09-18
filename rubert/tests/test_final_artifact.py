"""Финальный путь после сериализации: прогон протокола → export_final (как в train.py) → загрузка.

Модель — rubert-tiny2 на игрушечных данных, одна эпоха на CPU; сохранение и загрузка — настоящие.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from safetensors.torch import load, save_file

from rubert_clf import data, model_source, protocol
from rubert_clf.inference import ToxicityClassifier, is_peft_model

spec = importlib.util.spec_from_file_location("train_script", data.PROJECT_ROOT / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)

TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья",
         "Какой же ты тупой", "Согласен с автором", "Пошёл вон", "Отличная идея", "Бред несёшь, идиот", "Спасибо за помощь"]
LABELS = [1, 0] * 6
CANDIDATE = {"name": "toy", "model_dir": "models/rubert-tiny2", "max_length": 32, "truncation": "head", "lr": 5e-4,
             "epochs": 1, "batch_size": 4}


@pytest.fixture(scope="module")
def exported(tmp_path_factory, tokenizer):
    tmp = tmp_path_factory.mktemp("final")
    df = pd.DataFrame({data.ID_COL: range(len(TEXTS)), data.TEXT_COL: TEXTS, data.LABEL_COL: LABELS})
    parts = {"fit": df, "calibration": df, "validation": df, "test": df}
    run_spec = protocol.RunSpec(**{**CANDIDATE, "model_dir": str(model_source.MODEL_DIR)}, eval_batch_size=8, seed=0,
                                extra={"keep_model": True})
    import torch

    result = protocol.run(run_spec, parts, torch.device("cpu"), {}, log=lambda *_: None)
    model_dir, report = tmp / "model", tmp / "runs" / "toy__s0.json"
    selection = train_script.export_final(result, CANDIDATE, 0, data.PROJECT_ROOT / "configs" / "protocol.json",
                                          model_dir, report)
    return {"model_dir": model_dir, "report": report, "selection": selection, "result": result}


def copy_artifact(exported, tmp_path):
    """Своя копия артефакта и отчёта для порчи: selection.json ссылается на копию отчёта."""
    import shutil

    model_dir, report = tmp_path / "model", tmp_path / "toy__s0.json"
    shutil.copytree(exported["model_dir"], model_dir)
    shutil.copyfile(exported["report"], report)
    selection = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    selection["run"]["report"] = str(report)
    (model_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return model_dir, report


def test_loaded_artifact_reproduces_run_predictions_and_threshold(exported):
    clf = ToxicityClassifier.load(exported["model_dir"], "cpu")
    result = exported["result"]
    assert clf.threshold == result["threshold"] and clf.selection["epoch"] == result["selected_epoch"]
    assert clf.run_report["checkpoint"]["model_safetensors_sha256"] == data.sha256_file(exported["model_dir"] / "model.safetensors")
    probs = clf.predict_proba(TEXTS)
    np.testing.assert_allclose(probs, result["predictions"]["validation"], atol=2e-5)
    assert clf.predict(TEXTS).tolist() == (np.asarray(result["predictions"]["validation"]) >= clf.threshold).astype(int).tolist()


def test_changed_threshold_is_rejected(exported, tmp_path):
    model_dir, _ = copy_artifact(exported, tmp_path)
    selection = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    selection["threshold"] = 0.5
    (model_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="порог"):
        ToxicityClassifier.load(model_dir, "cpu")


def test_other_weights_are_rejected(exported, tmp_path):
    model_dir, _ = copy_artifact(exported, tmp_path)
    weights = load((model_dir / "model.safetensors").read_bytes())  # без отображения файла: на Windows его не перезаписать
    weights["classifier.bias"] = weights["classifier.bias"] + 1.0  # другие веса того же формата
    save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="SHA-256"):
        ToxicityClassifier.load(model_dir, "cpu")


def test_missing_or_incomplete_run_report_is_an_error(exported, tmp_path):
    model_dir, report = copy_artifact(exported, tmp_path)
    content = json.loads(report.read_text(encoding="utf-8"))
    del content["checkpoint"]
    report.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        ToxicityClassifier.load(model_dir, "cpu")
    report.unlink()
    with pytest.raises(FileNotFoundError, match="Отчёт прогона"):
        ToxicityClassifier.load(model_dir, "cpu")


LORA_CANDIDATE = {**CANDIDATE, "name": "toy_lora", "lr": 3e-3,
                  "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.1, "target_modules": ["query", "value"]}}


@pytest.fixture(scope="module")
def exported_lora(tmp_path_factory, tokenizer):
    pytest.importorskip("peft")
    import torch

    tmp = tmp_path_factory.mktemp("final_lora")
    df = pd.DataFrame({data.ID_COL: range(len(TEXTS)), data.TEXT_COL: TEXTS, data.LABEL_COL: LABELS})
    parts = {"fit": df, "calibration": df, "validation": df, "test": df}
    run_spec = protocol.RunSpec(**{**LORA_CANDIDATE, "model_dir": str(model_source.MODEL_DIR)}, eval_batch_size=8, seed=0,
                                extra={"keep_model": True})
    result = protocol.run(run_spec, parts, torch.device("cpu"), {}, log=lambda *_: None)
    # порог 0.5, а не калибровочный: на игрушечных данных он равен одной из вероятностей
    merged = protocol.measure_merged(result["model"], result["tokenizer"], TEXTS, run_spec,
                                     np.asarray(result["predictions"]["test"]), 0.5)
    model_dir, report = tmp / "model", tmp / "runs" / "toy_lora__s0.json"
    train_script.export_final(result, LORA_CANDIDATE, 0, data.PROJECT_ROOT / "configs" / "protocol.json",
                              model_dir, report)
    return {"model_dir": model_dir, "report": report, "result": result, "merged": merged}


def test_lora_artifact_is_adapter_only_and_points_to_registry(exported_lora):
    files = {p.name for p in exported_lora["model_dir"].iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors"} <= files and not files & {"model.safetensors", "README.md"}
    config = json.loads((exported_lora["model_dir"] / "adapter_config.json").read_text(encoding="utf-8"))
    assert (config["base_model_name_or_path"], config["revision"]) == (model_source.DEFAULT.model_id, model_source.DEFAULT.revision)
    assert "classifier" in str(config["modules_to_save"])  # голова классификатора сохранена вместе с адаптером
    checkpoint = exported_lora["result"]["checkpoint"]
    assert checkpoint["adapter_model_safetensors_sha256"] == data.sha256_file(exported_lora["model_dir"] / "adapter_model.safetensors")


def test_lora_artifact_reproduces_run_with_and_without_merge(exported_lora):
    result = exported_lora["result"]
    adapter = ToxicityClassifier.load(exported_lora["model_dir"], "cpu", merge=False)
    merged = ToxicityClassifier.load(exported_lora["model_dir"], "cpu")
    assert is_peft_model(adapter.model) and not is_peft_model(merged.model)
    assert merged.threshold == result["threshold"]
    np.testing.assert_allclose(adapter.predict_proba(TEXTS), result["predictions"]["validation"], atol=2e-5)
    np.testing.assert_allclose(merged.predict_proba(TEXTS), result["predictions"]["validation"], atol=2e-5)


def test_merged_measurement_matches_adapter(exported_lora):
    stats = exported_lora["merged"]
    assert stats["test_decision_flips_vs_adapter"] == 0 and stats["test_max_abs_diff_vs_adapter"] < 1e-4
    from transformers import AutoModelForSequenceClassification

    plain = AutoModelForSequenceClassification.from_pretrained(model_source.MODEL_DIR, num_labels=2)
    assert stats["parameters"] == sum(p.numel() for p in plain.parameters())  # после слияния — обычная модель


def test_lora_adapter_for_other_base_is_rejected(exported_lora, tmp_path):
    model_dir, _ = copy_artifact({"model_dir": exported_lora["model_dir"], "report": exported_lora["report"]}, tmp_path)
    config = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "ai-forever/ruBert-base"
    (model_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="адаптер обучен на"):
        ToxicityClassifier.load(model_dir, "cpu")
