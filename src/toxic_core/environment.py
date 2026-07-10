"""Версии библиотек фактического окружения расчёта; без импорта PyTorch."""

from importlib.metadata import version
import platform

COMMON = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
          "sklearn": "scikit-learn", "joblib": "joblib", "matplotlib": "matplotlib"}
NEURAL = {"torch": "torch", "transformers": "transformers", "tokenizers": "tokenizers",
          "huggingface_hub": "huggingface-hub", "safetensors": "safetensors",
          "peft": "peft", "accelerate": "accelerate"}


def capture(neural: bool = False) -> dict[str, str]:
    packages = COMMON | (NEURAL if neural else {})
    return {"python": platform.python_version(), **{name: version(package) for name, package in packages.items()}}
