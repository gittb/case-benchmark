"""pyannote-audio speaker embedding model wrapper.

pyannote-audio provides speaker diarization and embedding models.
The embedding model can be used standalone for speaker verification.

Model: pyannote/embedding
Architecture: TDNN-based
Training Data: VoxCeleb

Installation:
    pip install pyannote.audio

Note: pyannote models require accepting terms on HuggingFace Hub.
You may need to set HF_TOKEN environment variable.

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("pyannote")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path

import numpy as np

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


@register_model("pyannote")
@register_model("pyannote_embedding")
class PyannoteEmbeddingModel(EmbeddingModel):
    """pyannote-audio speaker embedding model."""

    def __init__(self, model_name: str = "pyannote/embedding"):
        """Initialize pyannote embedding model.

        Args:
            model_name: HuggingFace model ID
        """
        super().__init__()
        self.model_name = model_name
        self.model = None

    def load(self, device: str = "cpu") -> None:
        """Load pyannote embedding model."""
        try:
            from pyannote.audio import Model, Inference
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for this model. "
                "Install with: pip install pyannote.audio"
            )

        import torch

        # Load model from HuggingFace
        # Note: May require HF_TOKEN for gated models
        self.model = Model.from_pretrained(self.model_name)

        # Create inference pipeline
        self.inference = Inference(
            self.model,
            window="whole",  # Use whole file, not sliding window
        )

        # Move to device
        torch_device = torch.device(device)
        self.model.to(torch_device)

        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using pyannote."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio_path = str(audio_path)

        # pyannote inference returns embedding
        embedding = self.inference(audio_path)

        # Convert to numpy
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        elif hasattr(embedding, "data"):
            embedding = np.array(embedding.data)

        return np.array(embedding).flatten()

    @property
    def embedding_dim(self) -> int:
        return 512  # pyannote default

    @property
    def name(self) -> str:
        return "pyannote Embedding"
