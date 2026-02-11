"""Utility functions for CASE Benchmark evaluation.

This module provides:
- Audio loading with automatic resampling
- Trial list parsing
- Embedding I/O
- Path resolution for benchmark data
"""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


# Standard sample rate for all CASE Benchmark audio
SAMPLE_RATE = 16000


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    All CASE Benchmark audio is 16kHz mono. This function handles
    resampling from other sample rates and stereo-to-mono conversion.

    Args:
        path: Path to audio file (WAV, FLAC, or any format supported by soundfile)
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Tuple of (audio waveform as float32 numpy array in [-1, 1], sample rate)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio file can't be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {e}")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torchaudio

            audio_tensor = __import__("torch").from_numpy(audio)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio_tensor).numpy()
            sr = target_sr
        except ImportError:
            raise RuntimeError(
                f"Audio has sample rate {sr}, but target is {target_sr}. "
                "Install torchaudio for automatic resampling: pip install torchaudio"
            )

    return audio, sr


def parse_trial_line(line: str) -> tuple[int, str, str]:
    """Parse a single line from a trial list file.

    Trial list format:
        <label> <enrollment_path> <test_path>

    Where label is 1 for same speaker (target) and 0 for different speaker (impostor).

    Args:
        line: Single line from trial list file

    Returns:
        Tuple of (label, enrollment_path, test_path)

    Raises:
        ValueError: If line format is invalid
    """
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid trial line format: {line}")

    label = int(parts[0])
    enroll_path = parts[1]
    test_path = parts[2]

    return label, enroll_path, test_path


def load_trials(trials_path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    """Load a trial list file.

    Args:
        trials_path: Path to trial list text file

    Returns:
        Tuple of:
            - enrollment_paths: List of enrollment audio paths
            - test_paths: List of test audio paths
            - labels: NumPy array of labels (1 = target, 0 = impostor)
    """
    trials_path = Path(trials_path)
    if not trials_path.exists():
        raise FileNotFoundError(f"Trial list not found: {trials_path}")

    enrollment_paths = []
    test_paths = []
    labels = []

    with open(trials_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label, enroll, test = parse_trial_line(line)
                enrollment_paths.append(enroll)
                test_paths.append(test)
                labels.append(label)
            except ValueError:
                continue

    return enrollment_paths, test_paths, np.array(labels, dtype=np.int32)


def iter_trials(trials_path: str | Path) -> Iterator[tuple[int, str, str]]:
    """Iterate over trials in a trial list file.

    Memory-efficient alternative to load_trials for large trial lists.

    Args:
        trials_path: Path to trial list text file

    Yields:
        Tuples of (label, enrollment_path, test_path)
    """
    trials_path = Path(trials_path)

    with open(trials_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield parse_trial_line(line)
            except ValueError:
                continue


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    output_path: str | Path,
    format: str = "npz",
) -> None:
    """Save embeddings to disk.

    Args:
        embeddings: Dictionary mapping audio paths to embeddings
        output_path: Output file path
        format: Output format ('npz' for numpy archive, 'npy' for single array)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        np.savez_compressed(output_path, **embeddings)
    elif format == "npy":
        # Save as structured array
        paths = list(embeddings.keys())
        embs = np.array([embeddings[p] for p in paths])
        np.save(output_path, {"paths": paths, "embeddings": embs})
    else:
        raise ValueError(f"Unknown format: {format}")


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load embeddings from disk.

    Args:
        path: Path to embeddings file (.npz or .npy)

    Returns:
        Dictionary mapping audio paths to embeddings
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    elif path.suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
        paths = data["paths"]
        embs = data["embeddings"]
        return {p: e for p, e in zip(paths, embs)}
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def get_protocol_category(protocol_name: str) -> str:
    """Extract the category from a protocol name.

    Examples:
        "clean_clean" -> "clean"
        "clean_codec_gsm" -> "codec"
        "clean_playback_alaw_snr25_phone" -> "playback"

    Args:
        protocol_name: Name of the protocol (e.g., trial file stem)

    Returns:
        Category string (clean, codec, mic, noise, reverb, or playback)
    """
    parts = protocol_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def load_manifest(benchmark_dir: str | Path) -> dict:
    """Load the benchmark manifest.json file.

    The manifest contains metadata about the benchmark including:
    - Version information
    - Dataset source
    - Number of speakers
    - Protocol list

    Args:
        benchmark_dir: Path to benchmark directory

    Returns:
        Manifest dictionary
    """
    manifest_path = Path(benchmark_dir) / "benchmark" / "manifest.json"
    if not manifest_path.exists():
        # Try alternative location
        manifest_path = Path(benchmark_dir) / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in {benchmark_dir}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def verify_benchmark_integrity(benchmark_dir: str | Path) -> bool:
    """Verify benchmark data integrity using checksums.

    Args:
        benchmark_dir: Path to benchmark directory

    Returns:
        True if all files pass checksum verification
    """
    benchmark_dir = Path(benchmark_dir)
    checksum_path = benchmark_dir / "checksums.sha256"

    if not checksum_path.exists():
        # No checksums file, skip verification
        return True

    import hashlib

    with open(checksum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            expected_hash, filepath = parts
            full_path = benchmark_dir / filepath

            if not full_path.exists():
                print(f"Missing file: {filepath}")
                return False

            # Compute hash
            sha256_hash = hashlib.sha256()
            with open(full_path, "rb") as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    sha256_hash.update(chunk)

            if sha256_hash.hexdigest() != expected_hash:
                print(f"Checksum mismatch: {filepath}")
                return False

    return True
