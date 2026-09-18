"""Новый запуск и продолжение старого в scripts/experiments.py: обучение подменено, проверяется учёт."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from rubert_clf import data, experiment_run as er

spec = importlib.util.spec_from_file_location("experiments", data.PROJECT_ROOT / "scripts" / "experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PROTOCOL = {"seeds": [1, 2], "ready_classifiers": [],
            "candidates": [{"name": "a", "model_dir": "models/none", "epochs": 2},
                           {"name": "b", "model_dir": "models/none", "epochs": 2}]}


def toy_parts() -> dict:
    df = pd.DataFrame({data.ID_COL: range(6), data.TEXT_COL: ["x"] * 6, data.LABEL_COL: [0, 1] * 3})
    return {"fit": df, "calibration": df.iloc[:2], "validation": df, "test": df}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Протокол во временном каталоге, обучение — заглушка, которая записывает, что «обучено»."""
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(PROTOCOL), encoding="utf-8")
    trained, state = [], {"fail_on": None}

    def fake_run(spec, parts, device, masks):
        name = f"{spec.name}__s{spec.seed}"
        if name == state["fail_on"]:
            raise KeyboardInterrupt  # прерывание посреди запуска
        trained.append(name)
        return {"spec": {"name": spec.name, "seed": spec.seed, "epochs": spec.epochs},
                "epochs": [{"epoch": e, "train_loss": 0.3, "train_seconds": 1.5} for e in range(1, spec.epochs + 1)]}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "load_parts", toy_parts)
    monkeypatch.setattr(runner, "long_mask", lambda texts: np.zeros(len(texts), bool))
    monkeypatch.setattr(er, "model_dirs", lambda protocol, runs: [])

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["experiments.py", "--protocol", str(protocol_path), "--device", "cpu", *argv])
        runner.main()

    def resume(run_dir):
        monkeypatch.setattr(sys, "argv", ["experiments.py", "--resume", str(run_dir), "--device", "cpu"])
        runner.main()

    return {"main": main, "resume": resume, "trained": trained, "state": state, "protocol": protocol_path,
            "dir": tmp_path / "run"}


def test_new_run_trains_everything_and_passes_check(env):
    env["main"]("--out", str(env["dir"]))
    assert env["trained"] == ["a__s1", "a__s2", "b__s1", "b__s2"]
    assert er.verify(env["dir"], trained_in_last_session=True) == []
    report = er.read_json(env["dir"] / "runs" / "a__s1.json")
    assert report["run_id"] == "run" and report["session"] == 1


def test_new_run_refuses_existing_directory(env):
    env["main"]("--out", str(env["dir"]), "--only", "a")
    with pytest.raises(SystemExit, match="--resume"):
        env["main"]("--out", str(env["dir"]))
    assert env["trained"] == ["a__s1", "a__s2"]  # готовые отчёты не подхватываются молча


def test_resume_trains_only_missing_runs(env):
    env["state"]["fail_on"] = "b__s1"
    with pytest.raises(KeyboardInterrupt):
        env["main"]("--out", str(env["dir"]))
    assert er.verify(env["dir"]) == ["b__s1: прогон не выполнен", "b__s2: прогон не выполнен"]

    env["state"]["fail_on"] = None
    env["trained"].clear()
    env["resume"](env["dir"])
    assert env["trained"] == ["b__s1", "b__s2"]
    assert er.verify(env["dir"]) == []
    # обучение разнесено по двум сессиям — для workflow это не «всё обучено сейчас»
    assert er.verify(env["dir"], trained_in_last_session=True) == [
        "a__s1: выполнен не в последней сессии", "a__s2: выполнен не в последней сессии"]
    manifest = er.read_json(env["dir"] / er.MANIFEST)
    assert [s["mode"] for s in manifest["sessions"]] == ["new", "resume"]


def test_resume_refuses_changed_protocol(env):
    env["state"]["fail_on"] = "b__s1"
    with pytest.raises(KeyboardInterrupt):
        env["main"]("--out", str(env["dir"]))
    env["state"]["fail_on"] = None
    changed = {**PROTOCOL, "candidates": [{**PROTOCOL["candidates"][0], "lr": 1e-3}, PROTOCOL["candidates"][1]]}
    env["protocol"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="protocol.sha256"):
        env["resume"](env["dir"])


def test_resume_refuses_changed_data(env, monkeypatch):
    env["state"]["fail_on"] = "b__s1"
    with pytest.raises(KeyboardInterrupt):
        env["main"]("--out", str(env["dir"]))
    env["state"]["fail_on"] = None

    def other_calibration():
        parts = toy_parts()
        parts["calibration"] = parts["fit"].iloc[2:4]
        return parts

    monkeypatch.setattr(runner, "load_parts", other_calibration)
    with pytest.raises(SystemExit, match="calibration_source_rows_sha256"):
        env["resume"](env["dir"])


def test_resume_refuses_foreign_report(env):
    env["state"]["fail_on"] = "b__s1"
    with pytest.raises(KeyboardInterrupt):
        env["main"]("--out", str(env["dir"]))
    env["state"]["fail_on"] = None
    (env["dir"] / "runs" / "b__s1.json").write_text('{"run_id": "other"}', encoding="utf-8")  # подложенный отчёт
    with pytest.raises(SystemExit, match="b__s1.json не записан этим запуском"):
        env["resume"](env["dir"])


def test_resume_does_not_take_plan_options(env):
    env["main"]("--out", str(env["dir"]), "--only", "a")
    with pytest.raises(SystemExit):
        env["main"]("--resume", str(env["dir"]), "--only", "b")


def test_unknown_candidate_is_an_error(env):
    with pytest.raises(SystemExit):
        env["main"]("--out", str(env["dir"]), "--only", "nope")
    assert not env["dir"].exists()


def test_check_detects_epochs_without_training(env):
    env["main"]("--out", str(env["dir"]))
    path = env["dir"] / "runs" / "a__s2.json"
    report = er.read_json(path)
    report["epochs"] = report["epochs"][:1]
    er.write_json_atomic(path, report)
    assert er.verify(env["dir"]) == ["a__s2: эпох 1 из 2"]


def test_differences_lists_changed_keys():
    assert er.differences({"a": {"b": 1, "c": 2}}, {"a": {"b": 1, "c": 3}, "d": 4}) == ["a.c: 2 → 3", "d: None → 4"]
