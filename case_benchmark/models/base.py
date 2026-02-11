"""Base class for speaker embedding models.

All model wrappers must inherit from EmbeddingModel and implement the required
abstract methods. This provides a consistent interface for the CASE Benchmark
evaluation pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for speaker embedding models.

    Subclasses must implement:
        - load(): Load model weights and prepare for inference
        - extract_embedding(): Extract embedding from audio file
        - embedding_dim: Return the embedding dimension
        - name: Return the model name

    Example implementation:
        class MyModel(EmbeddingModel):
            def load(self, device: str = "cpu") -> None:
                self.model = load_my_model()
                self.model.to(device)
                self._device = device
                self._loaded = True

            def extract_embedding(self, audio_path: Path) -> np.ndarray:
                audio = load_audio(audio_path)
                embedding = self.model(audio)
                return embedding.numpy()

            @property
            def embedding_dim(self) -> int:
                return 192

            @property
            def name(self) -> str:
                return "my_model"
    """

    def __init__(self):
        self._loaded = False
        self._device = "cpu"

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model weights and prepare for inference.

        Args:
            device: Device to load model on ('cpu' or 'cuda')

        This method should:
            1. Download weights if necessary (from HuggingFace Hub, etc.)
            2. Initialize the model architecture
            3. Load pretrained weights
            4. Move model to specified device
            5. Set model to evaluation mode
            6. Set self._loaded = True
        """
        pass

    @abstractmethod
    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding from an audio file.

        Args:
            audio_path: Path to audio file (16kHz mono WAV)

        Returns:
            Speaker embedding as numpy array of shape (embedding_dim,)

        The implementation should:
            1. Load audio from file (16kHz expected)
            2. Apply any model-specific preprocessing
            3. Run forward pass to get embedding
            4. Return normalized embedding as numpy array
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the output embeddings."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable model name."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._loaded

    @property
    def device(self) -> str:
        """Return the device the model is loaded on."""
        return self._device

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.name}, dim={self.embedding_dim}, {status})"
