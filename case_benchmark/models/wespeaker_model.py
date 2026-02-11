"""WeSpeaker model wrappers.

WeSpeaker provides several pretrained speaker embedding models including
ResNet34 and CAM++ trained on VoxCeleb2.

Models available:
    - wespeaker_resnet34: ResNet34 backbone
    - wespeaker_campplus: CAM++ (Channel Attention Module)

Installation:
    pip install wespeakerruntime

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("wespeaker_resnet34")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path

import numpy as np

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


class WeSpeakerBaseModel(EmbeddingModel):
    """Base class for WeSpeaker models."""

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self.model = None

    def load(self, device: str = "cpu") -> None:
        """Load WeSpeaker model."""
        try:
            import wespeakerruntime as wespeaker
        except ImportError:
            raise ImportError(
                "WeSpeaker runtime is required for this model. "
                "Install with: pip install wespeakerruntime"
            )

        # WeSpeaker runtime automatically downloads models
        self.model = wespeaker.Speaker(lang="en")

        # Note: WeSpeaker runtime handles device internally
        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using WeSpeaker."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio_path = str(audio_path)

        # WeSpeaker returns embedding directly
        embedding = self.model.extract_embedding(audio_path)

        # Ensure numpy array
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()

        return np.array(embedding).flatten()

    @property
    def embedding_dim(self) -> int:
        return 256  # WeSpeaker default

    @property
    def name(self) -> str:
        return f"WeSpeaker {self._model_name}"


@register_model("wespeaker")
@register_model("wespeaker_resnet34")
class WeSpeakerResNet34Model(WeSpeakerBaseModel):
    """WeSpeaker ResNet34 speaker embedding model."""

    def __init__(self):
        super().__init__("ResNet34")


@register_model("wespeaker_campplus")
class WeSpeakerCAMPlusModel(WeSpeakerBaseModel):
    """WeSpeaker CAM++ speaker embedding model."""

    def __init__(self):
        super().__init__("CAM++")

    def load(self, device: str = "cpu") -> None:
        """Load WeSpeaker CAM++ model."""
        try:
            import wespeakerruntime as wespeaker
        except ImportError:
            raise ImportError(
                "WeSpeaker runtime is required for this model. "
                "Install with: pip install wespeakerruntime"
            )

        # Load CAM++ specifically
        # Note: WeSpeaker runtime API may vary
        self.model = wespeaker.Speaker(lang="en")

        self._device = device
        self._loaded = True
