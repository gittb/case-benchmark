"""NVIDIA NeMo TitaNet model wrapper.

NeMo provides TitaNet, a state-of-the-art speaker embedding model.
TitaNet uses a time-delay neural network with squeeze-and-excitation.

Model: nvidia/speakerverification_en_titanet_large
Architecture: TitaNet-L
Training Data: VoxCeleb + proprietary data
Embedding Dimension: 192

Installation:
    pip install nemo_toolkit[asr]

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("nemo_titanet")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path

import numpy as np

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


@register_model("nemo")
@register_model("nemo_titanet")
@register_model("titanet")
class NeMoTitaNetModel(EmbeddingModel):
    """NVIDIA NeMo TitaNet speaker embedding model."""

    def __init__(self, model_name: str = "nvidia/speakerverification_en_titanet_large"):
        """Initialize NeMo TitaNet model.

        Args:
            model_name: HuggingFace/NGC model ID
        """
        super().__init__()
        self.model_name = model_name
        self.model = None

    def load(self, device: str = "cpu") -> None:
        """Load NeMo TitaNet model."""
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError:
            raise ImportError(
                "NVIDIA NeMo is required for this model. "
                "Install with: pip install nemo_toolkit[asr]"
            )

        import torch

        # Load pretrained model
        self.model = EncDecSpeakerLabelModel.from_pretrained(self.model_name)

        # Move to device
        torch_device = torch.device(device)
        self.model.to(torch_device)
        self.model.eval()

        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using NeMo TitaNet."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        audio_path = str(audio_path)

        with torch.no_grad():
            # NeMo provides get_embedding method
            embedding = self.model.get_embedding(audio_path)

        # Convert to numpy
        embedding = embedding.cpu().numpy().flatten()

        return embedding

    @property
    def embedding_dim(self) -> int:
        return 192

    @property
    def name(self) -> str:
        return "NeMo TitaNet-L"
