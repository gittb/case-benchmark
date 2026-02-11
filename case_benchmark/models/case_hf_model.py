"""HuggingFace CASE Speaker Embedding model wrapper.

This model is specifically designed for carrier-agnostic speaker verification,
trained to be robust against various audio degradations (codecs, noise, reverb, etc.).

Model: bigstorm/case-speaker-embedding-v2-512
Architecture: ECAPA-TDNN with CASE augmentation
Training: VoxCeleb2 with 6-mode carrier augmentation pipeline
Embedding Dimension: 192
HuggingFace: https://huggingface.co/bigstorm/case-speaker-embedding-v2-512

Installation:
    pip install huggingface_hub torch torchaudio

Usage:
    >>> from case_benchmark.models import load_model
    >>> model = load_model("case_hf")
    >>> embedding = model.extract_embedding("audio.wav")
"""

from pathlib import Path
import sys
import importlib.util

import numpy as np
import torch

from case_benchmark.models.base import EmbeddingModel
from case_benchmark.models import register_model


@register_model("case_hf")
@register_model("case_hf_v2")
class CASEHuggingFaceModel(EmbeddingModel):
    """HuggingFace CASE Speaker Embedding model.

    This is the official CASE model designed for carrier-agnostic speaker
    verification. It produces 192-dimensional embeddings that are robust
    to audio degradations like codec compression, noise, and reverberation.
    """

    def __init__(
        self,
        model_name: str = "bigstorm/case-speaker-embedding-v2-512",
        sample_rate: int = 16000,
    ):
        """Initialize CASE HuggingFace model wrapper.

        Args:
            model_name: HuggingFace model ID
            sample_rate: Expected sample rate (16kHz)
        """
        super().__init__()
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.encoder = None

    def load(self, device: str = "cpu") -> None:
        """Load CASE model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for this model. "
                "Install with: pip install huggingface_hub"
            )

        # Download the model files
        model_py_path = hf_hub_download(self.model_name, "model.py")
        weights_path = hf_hub_download(self.model_name, "pytorch_model.bin")
        config_path = hf_hub_download(self.model_name, "config.json")

        # Load the model module dynamically
        spec = importlib.util.spec_from_file_location("case_model", model_py_path)
        case_module = importlib.util.module_from_spec(spec)
        sys.modules["case_model"] = case_module
        spec.loader.exec_module(case_module)

        # Get the model directory for from_pretrained
        model_dir = Path(weights_path).parent

        # Load the encoder
        self.encoder = case_module.CASESpeakerEncoder.from_pretrained(str(model_dir))

        # Move internal model to device
        if hasattr(self.encoder, 'model'):
            self.encoder.model.to(device)
            self.encoder.model.eval()

        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path | str) -> np.ndarray:
        """Extract speaker embedding using CASE model."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import soundfile as sf

        audio_path = Path(audio_path)

        # Load audio with soundfile (more reliable than torchaudio)
        waveform, sample_rate = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sample_rate != self.sample_rate:
            import torchaudio.functional as F
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            waveform_tensor = F.resample(waveform_tensor, sample_rate, self.sample_rate)
            waveform = waveform_tensor.squeeze().numpy()

        # Convert to tensor and pass to encoder
        waveform_tensor = torch.from_numpy(waveform).float().to(self._device)

        # Use the encoder's model directly
        with torch.no_grad():
            embedding = self.encoder.model(waveform_tensor.unsqueeze(0))

        # Ensure numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return np.array(embedding).flatten()

    @property
    def embedding_dim(self) -> int:
        return 192  # ECAPA-TDNN with 512 channels produces 192-dim embeddings

    @property
    def name(self) -> str:
        return "CASE HF v2-512"
