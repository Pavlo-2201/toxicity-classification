import numpy as np
import pandas as pd
import torch

from rubert_clf import data, model_source, protocol
from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.finetune import predict_proba

CPU = torch.device("cpu")
TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья",
         "Какой же ты тупой", "Согласен с автором", "Пошёл вон", "Отличная идея", "Бред несёшь, идиот", "Спасибо за помощь"]
LABELS = [1, 0] * 6


def toy_parts() -> dict:
    df = pd.DataFrame({data.ID_COL: range(len(TEXTS)), data.TEXT_COL: TEXTS, data.LABEL_COL: LABELS})
    return {"fit": df, "calibration": df, "validation": df, "test": df}


def test_keep_model_returns_weights_of_the_selected_epoch(tokenizer, monkeypatch):
    """Итоговая модель — веса эпохи с лучшим AP на validation, а не последней эпохи."""
    snapshots, calls = [], {"n": 0}
    original_train_epoch = protocol.train_epoch

    def recording_train_epoch(model, *args, **kwargs):
        stats = original_train_epoch(model, *args, **kwargs)
        snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return stats

    def scripted_ap(y, probs):  # AP по эпохам 0.5, 0.9, 0.7 только для validation (вызывается для неё одной)
        calls["n"] += 1
        return {"average_precision": [0.5, 0.9, 0.7][calls["n"] - 1], "log_loss": 0.0}

    monkeypatch.setattr(protocol, "train_epoch", recording_train_epoch)
    monkeypatch.setattr(protocol, "threshold_free", scripted_ap)
    spec = protocol.RunSpec(name="toy", model_dir=str(model_source.MODEL_DIR), max_length=32, lr=5e-4, epochs=3,
                            batch_size=4, eval_batch_size=8, seed=0, extra={"keep_model": True})
    result = protocol.run(spec, toy_parts(), CPU, {}, log=lambda *_: None)

    assert result["selected_epoch"] == 2 and len(snapshots) == 3
    model = result["model"]
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), snapshots[1][name]), name
    assert any(not torch.equal(snapshots[1][k], snapshots[2][k]) for k in snapshots[1])  # эпохи действительно разные

    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, 32)
    loader, order = make_length_sorted_dataloader(ds, tokenizer, 8)
    probs, _ = predict_proba(model, loader, CPU, order)
    np.testing.assert_allclose(probs, result["predictions"]["validation"], atol=1e-5)


def test_test_split_does_not_affect_epoch_or_threshold(tokenizer):
    """Эпоха — по validation, порог — по калибровочной части: перевёрнутые метки test ничего из этого не меняют."""
    df = toy_parts()["fit"]
    parts = {"fit": df, "calibration": df, "validation": df, "test": df}
    flipped = {**parts, "test": df.assign(**{data.LABEL_COL: 1 - df[data.LABEL_COL]})}
    spec = protocol.RunSpec(name="toy", model_dir=str(model_source.MODEL_DIR), max_length=32, lr=5e-4, epochs=2,
                            batch_size=4, eval_batch_size=8, seed=0)
    a = protocol.run(spec, parts, CPU, {}, log=lambda *_: None)
    b = protocol.run(spec, flipped, CPU, {}, log=lambda *_: None)
    assert a["selected_epoch"] == b["selected_epoch"] and a["threshold"] == b["threshold"]
    assert [e["validation"] for e in a["epochs"]] == [e["validation"] for e in b["epochs"]]
    np.testing.assert_array_equal(a["predictions"]["validation"], b["predictions"]["validation"])
    assert a["test"]["at_calibrated_threshold"] != b["test"]["at_calibrated_threshold"]  # test действительно другой
