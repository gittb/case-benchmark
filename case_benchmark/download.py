"""Download CASE Benchmark data from HuggingFace.

This module provides functionality to download the CASE Benchmark audio files
from HuggingFace Datasets. Audio files are stored as tar.gz archives per
condition type and automatically extracted after download.

Usage:
    from case_benchmark.download import download_benchmark

    # Download full benchmark
    download_benchmark("/path/to/output")

    # Download only specific conditions
    download_benchmark("/path/to/output", conditions=["clean", "codec"])

    # Or use CLI
    case-benchmark download --output-dir /path/to/output
"""

import tarfile
from pathlib import Path
from typing import Optional

# Archive manifest: maps archive names to their contents
ARCHIVES = {
    # VoxCeleb1-O dataset archives
    "voxceleb1_o_clean.tar.gz": ("voxceleb1_o", "clean"),
    "voxceleb1_o_codec.tar.gz": ("voxceleb1_o", "codec"),
    "voxceleb1_o_mic.tar.gz": ("voxceleb1_o", "mic"),
    "voxceleb1_o_noise.tar.gz": ("voxceleb1_o", "noise"),
    "voxceleb1_o_reverb.tar.gz": ("voxceleb1_o", "reverb"),
    "voxceleb1_o_playback.tar.gz": ("voxceleb1_o", "playback"),
    # LibriSpeech dataset archives
    "librispeech_clean.tar.gz": ("librispeech", "clean"),
    "librispeech_codec.tar.gz": ("librispeech", "codec"),
    "librispeech_mic.tar.gz": ("librispeech", "mic"),
    "librispeech_noise.tar.gz": ("librispeech", "noise"),
    "librispeech_reverb.tar.gz": ("librispeech", "reverb"),
    "librispeech_playback.tar.gz": ("librispeech", "playback"),
}

# All available conditions
ALL_CONDITIONS = ["clean", "codec", "mic", "noise", "reverb", "playback"]


def download_benchmark(
    output_dir: str | Path,
    repo_id: str = "bigstorm/case-benchmark",
    token: Optional[str] = None,
    conditions: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    keep_archives: bool = False,
) -> Path:
    """Download CASE Benchmark data from HuggingFace.

    Downloads audio archives and extracts them to the output directory.
    Trial lists are downloaded directly (not archived).

    Args:
        output_dir: Directory to save downloaded and extracted files
        repo_id: HuggingFace dataset repository ID
        token: Optional HuggingFace token for private repos
        conditions: List of conditions to download. Options:
            ["clean", "codec", "mic", "noise", "reverb", "playback"]
            If None, downloads all conditions.
        datasets: List of datasets to download. Options:
            ["voxceleb1_o", "librispeech"]
            If None, downloads both datasets.
        keep_archives: If True, keep .tar.gz files after extraction

    Returns:
        Path to downloaded benchmark directory

    Raises:
        ImportError: If huggingface_hub is not installed
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface-hub"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to all conditions and datasets
    if conditions is None:
        conditions = ALL_CONDITIONS
    if datasets is None:
        datasets = ["voxceleb1_o", "librispeech"]

    print(f"Downloading CASE Benchmark from {repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"Conditions: {conditions}")
    print(f"Datasets: {datasets}")
    print()

    # Step 1: Download trial lists and metadata (not archived)
    print("Downloading trial lists...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["trials/*", "*.json", "*.md"],
        token=token,
    )

    # Step 2: Download and extract audio archives
    archives_to_download = []
    for archive_name, (dataset, condition) in ARCHIVES.items():
        if dataset in datasets and condition in conditions:
            archives_to_download.append(archive_name)

    print(f"\nDownloading {len(archives_to_download)} audio archives...")

    for i, archive_name in enumerate(archives_to_download, 1):
        dataset, condition = ARCHIVES[archive_name]
        print(f"  [{i}/{len(archives_to_download)}] {archive_name}...")

        # Download archive
        archive_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"audio/{archive_name}",
            local_dir=output_dir,
            token=token,
        )

        # Extract archive
        print(f"    Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

        # Remove archive if not keeping
        if not keep_archives:
            Path(archive_path).unlink()

    # Clean up audio directory if archives were removed
    if not keep_archives:
        audio_dir = output_dir / "audio"
        if audio_dir.exists() and not any(audio_dir.iterdir()):
            audio_dir.rmdir()

    print()
    print(f"Download complete: {output_dir}")

    # Verify the download
    results = verify_download(output_dir)
    print(f"  Audio files: {results['audio_found']}")
    print(f"  Trial files: {results['trials_found']}")

    return output_dir


def verify_download(benchmark_dir: str | Path) -> dict:
    """Verify downloaded benchmark data.

    Args:
        benchmark_dir: Path to benchmark directory

    Returns:
        Dictionary with verification results including:
        - valid: Whether all expected files are present
        - trials_found: Number of trial files found
        - audio_found: Total audio files across all datasets
        - missing_trials: List of missing trial files
        - missing_audio_dirs: List of missing audio directories
    """
    benchmark_dir = Path(benchmark_dir)

    results = {
        "valid": True,
        "trials_found": 0,
        "audio_found": 0,
        "missing_trials": [],
        "missing_audio_dirs": [],
    }

    # Check trials directory
    trials_dir = benchmark_dir / "trials"
    if not trials_dir.exists():
        trials_dir = benchmark_dir / "benchmark" / "trials"

    # All 24 expected trial files
    expected_trials = [
        "clean_clean.txt",
        # Codecs (7)
        "clean_codec_alaw.txt",
        "clean_codec_gsm.txt",
        "clean_codec_mp3_32k.txt",
        "clean_codec_opus_6k.txt",
        "clean_codec_opus_12k.txt",
        "clean_codec_opus_24k.txt",
        "clean_codec_ulaw.txt",
        # Microphones (7)
        "clean_mic_conference_ceiling.txt",
        "clean_mic_headset_usb.txt",
        "clean_mic_laptop_internal.txt",
        "clean_mic_phone.txt",
        "clean_mic_smartphone_flagship.txt",
        "clean_mic_webcam_budget.txt",
        "clean_mic_webcam_quality.txt",
        # Noise (5)
        "clean_noise_snr_5.txt",
        "clean_noise_snr_10.txt",
        "clean_noise_snr_15.txt",
        "clean_noise_snr_20.txt",
        "clean_noise_snr_25.txt",
        # Reverb (1)
        "clean_reverb.txt",
        # Playback chains (3)
        "clean_playback_alaw_snr25_phone.txt",
        "clean_playback_gsm_snr20_webcam_budget.txt",
        "clean_playback_ulaw_snr15_laptop_internal.txt",
    ]

    if trials_dir.exists():
        trial_files = list(trials_dir.glob("*.txt"))
        results["trials_found"] = len(trial_files)

        for trial in expected_trials:
            if not (trials_dir / trial).exists():
                results["missing_trials"].append(trial)
                results["valid"] = False
    else:
        results["valid"] = False
        results["missing_trials"].append("trials directory")

    # Check audio directories (voxceleb1_o and librispeech)
    audio_dirs_to_check = ["voxceleb1_o", "librispeech"]
    expected_subdirs = ["clean", "codec", "mic", "noise", "reverb", "playback"]

    for audio_dir_name in audio_dirs_to_check:
        audio_dir = benchmark_dir / audio_dir_name
        if audio_dir.exists():
            # Count audio files
            audio_files = list(audio_dir.rglob("*.wav"))
            results["audio_found"] += len(audio_files)

            # Check expected subdirectories
            for subdir in expected_subdirs:
                subdir_path = audio_dir / subdir
                if not subdir_path.exists():
                    results["missing_audio_dirs"].append(f"{audio_dir_name}/{subdir}")
        else:
            results["missing_audio_dirs"].append(audio_dir_name)

    return results


def list_available_archives(
    repo_id: str = "bigstorm/case-benchmark",
    token: Optional[str] = None,
) -> list[dict]:
    """List available archives in the repository.

    Args:
        repo_id: HuggingFace dataset repository ID
        token: Optional HuggingFace token for private repos

    Returns:
        List of archive info dicts with keys: name, dataset, condition
    """
    archives = []
    for name, (dataset, condition) in ARCHIVES.items():
        archives.append({
            "name": name,
            "dataset": dataset,
            "condition": condition,
            "path": f"audio/{name}",
        })
    return archives


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CASE Benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_data"),
        help="Output directory",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        help="Conditions to download (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["voxceleb1_o", "librispeech"],
        help="Datasets to download (default: both)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep .tar.gz files after extraction",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing download instead of downloading",
    )

    args = parser.parse_args()

    if args.verify:
        results = verify_download(args.output_dir)
        print("Verification Results:")
        print(f"  Valid: {results['valid']}")
        print(f"  Trials found: {results['trials_found']}")
        print(f"  Audio files found: {results['audio_found']}")
        if results["missing_trials"]:
            print(f"  Missing trials: {results['missing_trials']}")
        if results["missing_audio_dirs"]:
            print(f"  Missing audio dirs: {results['missing_audio_dirs']}")
    else:
        download_benchmark(
            args.output_dir,
            conditions=args.conditions,
            datasets=args.datasets,
            keep_archives=args.keep_archives,
        )
