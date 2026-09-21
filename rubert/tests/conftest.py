import shutil

import pytest

from rubert_clf import data

FAILED = pytest.StashKey[bool]()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed:
        item.stash[FAILED] = True
    return report


@pytest.fixture(autouse=True)
def free_tmp_path_of_passed_test(request):
    """Каталог прошедшего теста удаляется сразу, а не в конце сессии.

    Тесты обучения пишут чекпойнты (last.pt с оптимизатором ~350 МБ); с удалением в конце сессии
    (tmp_path_retention_policy = "failed") они копятся до ~3.6 ГБ за прогон и на почти полном диске
    обрывают torch.save. Каталоги упавших тестов остаются для разбора, как и раньше.
    """
    path = request.getfixturevalue("tmp_path") if "tmp_path" in request.fixturenames else None
    yield
    if path is not None and not request.node.stash.get(FAILED, False):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def dataset():
    if not data.DATASET_FILE.exists():
        pytest.skip("данные не подготовлены: toxic prepare")
    return data.load_dataset()


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    from rubert_clf import model_source

    if not (model_source.MODEL_DIR / "config.json").exists():
        pytest.skip("модель не скачана: toxic prepare rubert")
    return AutoTokenizer.from_pretrained(model_source.MODEL_DIR)


@pytest.fixture(scope="session")
def encoder(tokenizer):
    from transformers import AutoModel

    from rubert_clf import model_source

    return AutoModel.from_pretrained(model_source.MODEL_DIR).eval()
