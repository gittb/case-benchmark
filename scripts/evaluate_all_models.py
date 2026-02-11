#!/usr/bin/env python3
"""Evaluate all models on the CASE Benchmark.

This script runs evaluation on multiple speaker embedding models and
saves results in a standardized format for leaderboard generation.

Usage:
    python scripts/evaluate_all_models.py --benchmark-dir /hdd_nas/datasets/case/benchmark
    python scripts/evaluate_all_models.py --models speechbrain resemblyzer
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


# ============================================================================
# Metrics
# ============================================================================

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    try:
        diff = fpr - fnr
        idx = np.argmin(np.abs(diff))

        if idx > 0 and idx < len(fpr) - 1:
            eer = brentq(
                lambda x: interp1d(fpr, fnr)(x) - x,
                fpr[max(0, idx - 1)],
                fpr[min(len(fpr) - 1, idx + 1)],
            )
        else:
            eer = fpr[idx]
        threshold = thresholds[idx]
    except (ValueError, IndexError):
        idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
) -> tuple[float, float]:
    """Compute minimum Detection Cost Function."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    dcf = fnr * p_target + fpr * (1 - p_target)
    min_idx = np.argmin(dcf)
    min_dcf = dcf[min_idx]
    threshold = thresholds[min_idx]

    c_default = min(p_target, 1 - p_target)
    min_dcf_norm = min_dcf / c_default

    return float(min_dcf_norm), float(threshold)


def compute_cosine_scores(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between embedding pairs."""
    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    norm1 = np.maximum(norm1, 1e-8)
    norm2 = np.maximum(norm2, 1e-8)
    emb1_norm = emb1 / norm1
    emb2_norm = emb2 / norm2
    return np.sum(emb1_norm * emb2_norm, axis=1)


# ============================================================================
# Model Wrappers
# ============================================================================

class BaseModel:
    """Base class for speaker embedding models."""

    def __init__(self):
        self._loaded = False
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        raise NotImplementedError

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError


class SpeechBrainModel(BaseModel):
    """SpeechBrain ECAPA-TDNN model."""

    def load(self, device: str = "cpu") -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        audio = self.classifier.load_audio(str(audio_path))
        embedding = self.classifier.encode_batch(audio)
        return embedding.squeeze().cpu().numpy()

    @property
    def name(self) -> str:
        return "SpeechBrain ECAPA-TDNN"

    @property
    def embedding_dim(self) -> int:
        return 192


class ResemblyzerModel(BaseModel):
    """Resemblyzer GE2E model."""

    def load(self, device: str = "cpu") -> None:
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder(device=device)
        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(audio_path)
        embedding = self.encoder.embed_utterance(wav)
        return np.array(embedding).flatten()

    @property
    def name(self) -> str:
        return "Resemblyzer"

    @property
    def embedding_dim(self) -> int:
        return 256


class WeSpeakerModel(BaseModel):
    """WeSpeaker model (ONNX-based, supports GPU with onnxruntime-gpu).

    Note: wespeakerruntime doesn't expose ONNX providers, so we create
    our own InferenceSession with explicit GPU support.
    Uses soundfile + torch for feature extraction to avoid torchaudio issues.
    """

    def load(self, device: str = "cpu") -> None:
        import onnxruntime as ort
        from wespeakerruntime.hub import Hub

        # Get the ONNX model path
        onnx_path = Hub.get_model_by_lang("en")

        # Configure session options
        so = ort.SessionOptions()
        so.inter_op_num_threads = 4
        so.intra_op_num_threads = 4

        # Set up providers based on device
        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("    WeSpeaker: Using CUDA execution provider")
        else:
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                print("    WeSpeaker: CUDA requested but not available, using CPU")
                print("    Install GPU support: pip install onnxruntime-gpu")

        # Create session with explicit providers
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self._device = device
        self._loaded = True

    def _compute_fbank(self, wav_path: str, target_sr: int = 16000,
                       num_mel_bins: int = 80, frame_length: int = 25,
                       frame_shift: int = 10, cmn: bool = True):
        """Extract fbank features using soundfile + torch (GPU-accelerated)."""
        import torch
        import torch.nn.functional as F
        import torchaudio.compliance.kaldi as kaldi

        # Load audio with soundfile (more robust than torchaudio)
        audio, sample_rate = sf.read(wav_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Resample if needed
        if sample_rate != target_sr:
            audio = torch.from_numpy(audio).unsqueeze(0)
            new_len = int(len(audio[0]) * target_sr / sample_rate)
            audio = F.interpolate(audio.unsqueeze(0), size=new_len, mode='linear', align_corners=False)
            audio = audio.squeeze().numpy()
            sample_rate = target_sr

        # Convert to torch tensor and scale (same as wespeaker)
        waveform = torch.from_numpy(audio).unsqueeze(0) * (1 << 15)

        # Move to GPU for faster fbank computation if available
        if self._device == "cuda" and torch.cuda.is_available():
            waveform = waveform.cuda()

        # Compute fbank using torchaudio.compliance.kaldi (GPU-accelerated)
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=0.0,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)

        # Move back to CPU for numpy conversion
        mat = mat.cpu().numpy()
        if cmn:
            mat = mat - np.mean(mat, axis=0)
        return mat

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        feats = self._compute_fbank(str(audio_path))
        feats = np.expand_dims(feats, 0).astype(np.float32)
        embeddings = self.session.run(output_names=['embs'],
                                      input_feed={'feats': feats})
        return np.array(embeddings[0]).flatten()

    @property
    def name(self) -> str:
        return "WeSpeaker ResNet34"

    @property
    def embedding_dim(self) -> int:
        return 256


class WeSpeakerCAMPPModel(BaseModel):
    """WeSpeaker CAM++ model (ONNX-based).

    CAM++ achieves 0.71% EER on VoxCeleb1-O, competitive with SOTA.
    Uses the same feature extraction as WeSpeaker ResNet34.
    """

    def load(self, device: str = "cpu") -> None:
        import onnxruntime as ort
        import urllib.request
        import os

        # Download CAM++ model if not exists
        cache_dir = os.path.expanduser('~/.cache/wespeaker')
        os.makedirs(cache_dir, exist_ok=True)
        onnx_path = os.path.join(cache_dir, 'voxceleb_CAM++_LM.onnx')

        if not os.path.exists(onnx_path):
            cam_url = 'https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_CAM++_LM.onnx'
            print(f"    Downloading CAM++ model...")
            urllib.request.urlretrieve(cam_url, onnx_path)

        # Configure session options
        so = ort.SessionOptions()
        so.inter_op_num_threads = 4
        so.intra_op_num_threads = 4

        # Set up providers based on device
        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("    CAM++: Using CUDA execution provider")
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self._device = device
        self._loaded = True

    def _compute_fbank(self, wav_path: str, target_sr: int = 16000,
                       num_mel_bins: int = 80, frame_length: int = 25,
                       frame_shift: int = 10, cmn: bool = True):
        """Extract fbank features using soundfile + torch."""
        import torch
        import torch.nn.functional as F
        import torchaudio.compliance.kaldi as kaldi

        audio, sample_rate = sf.read(wav_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sample_rate != target_sr:
            audio = torch.from_numpy(audio).unsqueeze(0)
            new_len = int(len(audio[0]) * target_sr / sample_rate)
            audio = F.interpolate(audio.unsqueeze(0), size=new_len, mode='linear', align_corners=False)
            audio = audio.squeeze().numpy()
            sample_rate = target_sr

        waveform = torch.from_numpy(audio).unsqueeze(0) * (1 << 15)

        if self._device == "cuda" and torch.cuda.is_available():
            waveform = waveform.cuda()

        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=0.0,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)

        mat = mat.cpu().numpy()
        if cmn:
            mat = mat - np.mean(mat, axis=0)
        return mat

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        feats = self._compute_fbank(str(audio_path))
        feats = np.expand_dims(feats, 0).astype(np.float32)
        embeddings = self.session.run(output_names=['embs'],
                                      input_feed={'feats': feats})
        return np.array(embeddings[0]).flatten()

    @property
    def name(self) -> str:
        return "WeSpeaker CAM++"

    @property
    def embedding_dim(self) -> int:
        return 512


class PyannoteModel(BaseModel):
    """pyannote-audio embedding model.

    Uses soundfile for audio loading to avoid torchcodec/torchaudio issues.
    """

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # pyannote expects 16kHz

    def load(self, device: str = "cpu") -> None:
        from pyannote.audio import Model

        self.model = Model.from_pretrained("pyannote/embedding")
        self.model.to(torch.device(device))
        self.model.eval()
        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        # Load audio with soundfile (avoids torchcodec issues)
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sample_rate != self.sample_rate:
            import torch.nn.functional as F
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            new_len = int(len(audio) * self.sample_rate / sample_rate)
            audio_tensor = F.interpolate(
                audio_tensor.unsqueeze(0), size=new_len, mode='linear', align_corners=False
            )
            audio = audio_tensor.squeeze().numpy()

        # Convert to tensor with shape (batch=1, channels=1, samples)
        waveform = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self._device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(waveform)

        # Handle output format
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return np.array(embedding).flatten()

    @property
    def name(self) -> str:
        return "pyannote Embedding"

    @property
    def embedding_dim(self) -> int:
        return 512


class NeMoTitaNetModel(BaseModel):
    """NVIDIA NeMo TitaNet model."""

    def load(self, device: str = "cpu") -> None:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.model = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.model.to(torch.device(device))
        self.model.eval()
        self._device = device
        self._loaded = True

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        with torch.no_grad():
            embedding = self.model.get_embedding(str(audio_path))
        return embedding.cpu().numpy().flatten()

    @property
    def name(self) -> str:
        return "NeMo TitaNet-L"

    @property
    def embedding_dim(self) -> int:
        return 192


class CASEHFModel(BaseModel):
    """HuggingFace CASE Speaker Embedding model.

    Model: bigstorm/case-speaker-embedding-v2-512
    Embedding Dimension: 192
    """

    def __init__(self, model_name: str = "bigstorm/case-speaker-embedding-v2-512"):
        super().__init__()
        self.model_name = model_name
        self.sample_rate = 16000

    def load(self, device: str = "cpu") -> None:
        from huggingface_hub import hf_hub_download
        import importlib.util

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

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        # Load audio with soundfile (more reliable than torchaudio)
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sample_rate != self.sample_rate:
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_tensor = F.resample(audio_tensor, sample_rate, self.sample_rate)
            audio = audio_tensor.squeeze().numpy()

        # Convert to tensor and pass to encoder
        audio_tensor = torch.from_numpy(audio).float().to(self._device)

        # Use the encoder's model directly
        with torch.no_grad():
            embedding = self.encoder.model(audio_tensor.unsqueeze(0))

        # Ensure numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        return np.array(embedding).flatten()

    @property
    def name(self) -> str:
        return "CASE HF v2-512"

    @property
    def embedding_dim(self) -> int:
        return 192


# Model registry
MODEL_REGISTRY = {
    "speechbrain": SpeechBrainModel,
    "resemblyzer": ResemblyzerModel,
    "wespeaker": WeSpeakerModel,
    # "wespeaker_campp": WeSpeakerCAMPPModel,  # ONNX export has issues
    "pyannote": PyannoteModel,
    "nemo": NeMoTitaNetModel,
    "case_hf": CASEHFModel,
}


def get_model(name: str) -> BaseModel:
    """Get model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()


# ============================================================================
# Evaluation
# ============================================================================

def load_trials(trials_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Load trial list."""
    enroll_paths = []
    test_paths = []
    labels = []

    with open(trials_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.append(int(parts[0]))
                enroll_paths.append(parts[1])
                test_paths.append(parts[2])

    return enroll_paths, test_paths, np.array(labels)


def evaluate_protocol(
    model: BaseModel,
    trials_path: Path,
    benchmark_dir: Path,
    embedding_cache: dict,
    show_progress: bool = True,
) -> dict:
    """Evaluate a single protocol."""
    enroll_paths, test_paths, labels = load_trials(trials_path)

    if len(enroll_paths) == 0:
        return {"eer": float("nan"), "min_dcf": float("nan"), "num_trials": 0}

    scores = []
    valid_labels = []

    iterator = zip(enroll_paths, test_paths, labels)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"  {trials_path.stem}", leave=False)

    for enroll_path, test_path, label in iterator:
        enroll_full = benchmark_dir / enroll_path
        test_full = benchmark_dir / test_path

        if not enroll_full.exists() or not test_full.exists():
            continue

        try:
            # Get embeddings (with caching)
            if str(enroll_full) not in embedding_cache:
                embedding_cache[str(enroll_full)] = model.extract_embedding(enroll_full)
            if str(test_full) not in embedding_cache:
                embedding_cache[str(test_full)] = model.extract_embedding(test_full)

            emb1 = embedding_cache[str(enroll_full)]
            emb2 = embedding_cache[str(test_full)]

            # Compute score
            score = compute_cosine_scores(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1),
            )[0]

            scores.append(score)
            valid_labels.append(label)
        except Exception as e:
            continue

    if len(scores) == 0:
        return {"eer": float("nan"), "min_dcf": float("nan"), "num_trials": 0}

    scores = np.array(scores)
    valid_labels = np.array(valid_labels)

    eer, eer_thresh = compute_eer(scores, valid_labels)
    min_dcf, dcf_thresh = compute_min_dcf(scores, valid_labels)

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "min_dcf": min_dcf,
        "min_dcf_threshold": dcf_thresh,
        "num_trials": len(scores),
        "num_target": int(np.sum(valid_labels)),
        "num_impostor": int(len(valid_labels) - np.sum(valid_labels)),
    }


def collect_unique_audio_files(trials_dir: Path, benchmark_dir: Path) -> set[Path]:
    """Collect all unique audio files across all protocols."""
    unique_files = set()
    for trials_path in trials_dir.glob("*.txt"):
        with open(trials_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    unique_files.add(benchmark_dir / parts[1])
                    unique_files.add(benchmark_dir / parts[2])
    return unique_files


def pre_extract_embeddings(
    model: BaseModel,
    audio_files: set[Path],
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Pre-extract embeddings for all audio files."""
    cache = {}
    files = [f for f in audio_files if f.exists()]

    iterator = files
    if show_progress:
        iterator = tqdm(files, desc="  Pre-extracting embeddings")

    for audio_path in iterator:
        try:
            cache[str(audio_path)] = model.extract_embedding(audio_path)
        except Exception as e:
            continue

    return cache


def evaluate_model(
    model: BaseModel,
    benchmark_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    preextract: bool = True,
) -> dict:
    """Evaluate a model on the full CASE Benchmark."""
    trials_dir = benchmark_dir / "trials"

    if not trials_dir.exists():
        raise FileNotFoundError(f"Trials directory not found: {trials_dir}")

    # Load model
    print(f"\nLoading model: {model.name}")
    start_time = time.time()
    model.load(device)
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")

    # Find all trial files
    trial_files = sorted(trials_dir.glob("*.txt"))
    print(f"  Protocols: {len(trial_files)}")

    # Evaluate each protocol
    all_results = {}
    embedding_cache = {}

    eval_start = time.time()

    # Pre-extract all embeddings (much faster than on-the-fly)
    if preextract:
        print("  Collecting unique audio files...")
        unique_files = collect_unique_audio_files(trials_dir, benchmark_dir)
        print(f"  Found {len(unique_files)} unique audio files")
        embedding_cache = pre_extract_embeddings(model, unique_files)
        print(f"  Pre-extracted {len(embedding_cache)} embeddings")

    for trials_path in tqdm(trial_files, desc="Protocols"):
        protocol_name = trials_path.stem
        results = evaluate_protocol(
            model=model,
            trials_path=trials_path,
            benchmark_dir=benchmark_dir,
            embedding_cache=embedding_cache,
        )
        all_results[protocol_name] = results

        if not np.isnan(results["eer"]):
            print(f"    {protocol_name}: EER={results['eer']*100:.2f}%")

    eval_time = time.time() - eval_start
    print(f"  Evaluation completed in {eval_time:.1f}s")

    # Compute CASE-Score
    clean_eer = all_results.get("clean_clean", {}).get("eer", 0.01)
    if clean_eer == 0 or np.isnan(clean_eer):
        clean_eer = 0.01

    # Group by category
    from collections import defaultdict
    category_eers = defaultdict(list)

    for name, res in all_results.items():
        eer = res.get("eer")
        if eer is None or np.isnan(eer):
            continue
        parts = name.split("_")
        category = parts[1] if len(parts) >= 2 else parts[0]
        category_eers[category].append(eer)

    # Compute weighted average (V1 - legacy)
    contributions = {}
    total_weight = 0
    weighted_sum = 0

    # V2 metrics accumulators
    weighted_eer_sum = 0
    weighted_degradation_sum = 0

    for category, eers in category_eers.items():
        avg_eer = np.mean(eers)
        weight = 1.0

        if category != "clean":
            normalized = avg_eer / clean_eer
            degradation = avg_eer - clean_eer
        else:
            normalized = avg_eer
            degradation = 0.0

        contributions[category] = {
            "avg_eer": float(avg_eer),
            "normalized_eer": float(normalized),
            "degradation": float(degradation),
            "weight": weight,
            "n_protocols": len(eers),
        }

        weighted_sum += weight * normalized
        weighted_eer_sum += weight * avg_eer
        weighted_degradation_sum += weight * degradation
        total_weight += weight

    case_score = weighted_sum / total_weight if total_weight > 0 else float("nan")
    case_score_absolute = weighted_eer_sum / total_weight if total_weight > 0 else float("nan")
    degradation_factor = weighted_degradation_sum / total_weight if total_weight > 0 else float("nan")

    # Save results
    output = {
        "model_name": model.name,
        "case_score": case_score,
        "case_score_v2": {
            "case_score_absolute": case_score_absolute,
            "degradation_factor": degradation_factor,
            "clean_eer": clean_eer,
        },
        "config": {
            "benchmark_dir": str(benchmark_dir),
            "device": device,
            "eval_time_seconds": eval_time,
        },
        "contributions": contributions,
        "protocol_results": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe_name = model.name.lower().replace(" ", "_").replace("-", "_")
    output_path = output_dir / f"{model_safe_name}_results.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  CASE-Score (v1): {case_score:.3f}")
    print(f"  CASE-Score v2:")
    print(f"    Absolute EER: {case_score_absolute*100:.2f}%")
    print(f"    Degradation:  {degradation_factor*100:.2f}%")
    print(f"  Results saved to: {output_path}")

    return output


def print_summary(results: list[dict]) -> None:
    """Print summary table of all results."""
    print("\n" + "=" * 100)
    print("CASE Benchmark Results Summary")
    print("=" * 100)

    print(f"\n{'Model':<25} {'Absolute':>10} {'Degrad.':>10} {'Clean':>8} {'Codec':>8} {'Reverb':>8} {'Playback':>10}")
    print("-" * 100)

    # Sort by absolute EER (lower is better)
    def sort_key(r):
        v2 = r.get("case_score_v2", {})
        return v2.get("case_score_absolute", float("inf"))

    for r in sorted(results, key=sort_key):
        name = r.get("model_name", "Unknown")[:23]
        v2 = r.get("case_score_v2", {})
        abs_eer = v2.get("case_score_absolute", float("nan"))
        deg = v2.get("degradation_factor", float("nan"))

        contrib = r.get("contributions", {})
        clean = contrib.get("clean", {}).get("avg_eer", float("nan"))
        codec = contrib.get("codec", {}).get("avg_eer", float("nan"))
        reverb = contrib.get("reverb", {}).get("avg_eer", float("nan"))
        playback = contrib.get("playback", {}).get("avg_eer", float("nan"))

        abs_str = f"{abs_eer*100:.2f}%" if not np.isnan(abs_eer) else "N/A"
        deg_str = f"{deg*100:+.2f}%" if not np.isnan(deg) else "N/A"
        clean_str = f"{clean*100:.2f}%" if not np.isnan(clean) else "N/A"
        codec_str = f"{codec*100:.2f}%" if not np.isnan(codec) else "N/A"
        reverb_str = f"{reverb*100:.2f}%" if not np.isnan(reverb) else "N/A"
        playback_str = f"{playback*100:.2f}%" if not np.isnan(playback) else "N/A"

        print(f"{name:<25} {abs_str:>10} {deg_str:>10} {clean_str:>8} {codec_str:>8} {reverb_str:>8} {playback_str:>10}")

    print("-" * 100)
    print()


def get_best_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f}GB)")
        return "cuda"
    else:
        print("No GPU detected, using CPU")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on CASE Benchmark")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("./benchmark"),
        help="Path to benchmark directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=f"Models to evaluate (default: all). Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (default), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device == "auto":
        args.device = get_best_device()

    if args.list_models:
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(f"  - {name}")
        return

    # Determine which models to evaluate
    models_to_eval = args.models or list(MODEL_REGISTRY.keys())

    print("=" * 80)
    print("CASE Benchmark Evaluation")
    print("=" * 80)
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Models: {models_to_eval}")
    print()

    all_results = []

    for model_name in models_to_eval:
        print(f"\n{'='*40}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*40}")

        try:
            model = get_model(model_name)
            results = evaluate_model(
                model=model,
                benchmark_dir=args.benchmark_dir,
                output_dir=args.output_dir,
                device=args.device,
            )
            all_results.append(results)
        except ImportError as e:
            print(f"  Skipping {model_name}: {e}")
            print(f"  Install with: pip install case-benchmark[{model_name}]")
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
