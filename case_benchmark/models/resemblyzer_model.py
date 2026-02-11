"""Resemblyzer speaker embedding model wrapper.

Resemblyzer is a lightweight speaker embedding model based on GE2E loss.
It's popular for its simplicity and ease of use.

Model: Resemblyzer
Architecture: 3-layer LSTM + FC
Training Data: VoxCeleb
Embedding Dimension: 256

Installation:
    pip install resemblyzer

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("resemblyzer")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path

import numpy as np

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


@register_model("resemblyzer")
class ResemblyzerModel(EmbeddingModel):
    """Resemblyzer speaker embedding model."""

    def __init__(self):
        super().__init__()
        self.encoder = None

    def load(self, device: str = "cpu") -> None:
        """Load Resemblyzer model."""
        try:
            from resemblyzer import VoiceEncoder
        except ImportError:
            raise ImportError(
                "Resemblyzer is required for this model. "
                "Install with: pip install resemblyzer"
            )

        # Load voice encoder
        # Resemblyzer uses 'cuda' or 'cpu' string
        self.encoder = VoiceEncoder(device=device)

        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using Resemblyzer."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            from resemblyzer import preprocess_wav
        except ImportError:
            raise ImportError("Resemblyzer is required. Install with: pip install resemblyzer")

        audio_path = Path(audio_path)

        # Load and preprocess audio
        wav = preprocess_wav(audio_path)

        # Extract embedding
        embedding = self.encoder.embed_utterance(wav)

        return np.array(embedding).flatten()

    @property
    def embedding_dim(self) -> int:
        return 256

    @property
    def name(self) -> str:
        return "Resemblyzer"
