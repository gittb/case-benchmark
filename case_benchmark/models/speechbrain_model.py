"""SpeechBrain ECAPA-TDNN model wrapper.

SpeechBrain provides a pretrained ECAPA-TDNN model trained on VoxCeleb1+2.
This is one of the most popular open-source speaker embedding models.

Model: spkrec-ecapa-voxceleb
Architecture: ECAPA-TDNN
Training Data: VoxCeleb1 + VoxCeleb2
Embedding Dimension: 192
Paper: https://arxiv.org/abs/2005.07143

Installation:
    pip install speechbrain

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("speechbrain")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path

import numpy as np

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


@register_model("speechbrain")
@register_model("speechbrain_ecapa")
class SpeechBrainECAPAModel(EmbeddingModel):
    """SpeechBrain ECAPA-TDNN speaker embedding model.

    This wrapper uses the official SpeechBrain pretrained model from HuggingFace:
    speechbrain/spkrec-ecapa-voxceleb
    """

    def __init__(self, model_source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        """Initialize SpeechBrain ECAPA-TDNN wrapper.

        Args:
            model_source: HuggingFace model ID or local path
        """
        super().__init__()
        self.model_source = model_source
        self.classifier = None

    def load(self, device: str = "cpu") -> None:
        """Load SpeechBrain ECAPA-TDNN model."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for this model. "
                "Install with: pip install speechbrain"
            )

        # Load pretrained model from HuggingFace
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_source,
            run_opts={"device": device},
        )

        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using SpeechBrain."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio_path = Path(audio_path)

        # SpeechBrain handles audio loading internally
        embedding = self.classifier.encode_batch(
            self.classifier.load_audio(str(audio_path))
        )

        # Convert to numpy and flatten
        embedding = embedding.squeeze().cpu().numpy()

        return embedding

    @property
    def embedding_dim(self) -> int:
        return 192

    @property
    def name(self) -> str:
        return "SpeechBrain ECAPA-TDNN"
