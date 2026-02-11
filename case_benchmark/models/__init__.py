"""Model wrappers for CASE Benchmark evaluation.

This module provides a unified interface for evaluating different speaker embedding
models on the CASE Benchmark. Each wrapper handles model loading, preprocessing,
and embedding extraction for a specific model/framework.

Supported Models:
    - SpeechBrain ECAPA-TDNN (speechbrain)
    - WeSpeaker ResNet34/CAM++ (wespeaker)
    - pyannote-audio (pyannote)
    - NVIDIA NeMo TitaNet (nemo)
    - Resemblyzer (resemblyzer)

Usage:
    >>> from case_benchmark.models import load_model, list_available_models
    >>> print(list_available_models())
    ['speechbrain', 'wespeaker_resnet34', 'wespeaker_campplus', ...]
    >>> model = load_model("speechbrain")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from case_benchmark.models.base import EmbeddingModel

# Registry of available models
_MODEL_REGISTRY: dict[str, type[EmbeddingModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def list_available_models() -> list[str]:
    """List all available model names."""
    # Import all model modules to populate registry
    _import_all_models()
    return sorted(_MODEL_REGISTRY.keys())


def load_model(
    name: str,
    device: str = "cpu",
    **kwargs,
) -> EmbeddingModel:
    """Load a speaker embedding model by name.

    Args:
        name: Model name (see list_available_models())
        device: Device to load model on ('cpu' or 'cuda')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized EmbeddingModel instance

    Raises:
        ValueError: If model name is not recognized
        ImportError: If required dependencies are not installed

    Example:
        >>> model = load_model("speechbrain", device="cuda")
        >>> model.load()
        >>> embedding = model.extract_embedding("audio.wav")
    """
    _import_all_models()

    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {name}. Available models: {available}\n"
            "Make sure to install the required dependencies. See pyproject.toml [project.optional-dependencies]"
        )

    model_cls = _MODEL_REGISTRY[name]
    model = model_cls(**kwargs)
    model.load(device)
    return model


def _import_all_models():
    """Import all model modules to populate the registry."""
    try:
        from case_benchmark.models import speechbrain_model
    except ImportError:
        pass

    try:
        from case_benchmark.models import wespeaker_model
    except ImportError:
        pass

    try:
        from case_benchmark.models import pyannote_model
    except ImportError:
        pass

    try:
        from case_benchmark.models import nemo_model
    except ImportError:
        pass

    try:
        from case_benchmark.models import resemblyzer_model
    except ImportError:
        pass

    try:
        from case_benchmark.models import case_hf_model
    except ImportError:
        pass


__all__ = [
    "EmbeddingModel",
    "load_model",
    "list_available_models",
    "register_model",
]
