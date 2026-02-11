#!/usr/bin/env python3
"""Download CASE Benchmark audio files from HuggingFace.

Usage:
    python scripts/download_audio.py --output-dir /path/to/output
    python scripts/download_audio.py --output-dir ./benchmark_data --verify
"""

import argparse
from pathlib import Path

from case_benchmark.download import download_benchmark, verify_download


def main():
    parser = argparse.ArgumentParser(
        description="Download CASE Benchmark audio files from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_data"),
        help="Output directory for downloaded files (default: benchmark_data/)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download only audio files (skip metadata)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing download instead of downloading",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (for private repos)",
    )

    args = parser.parse_args()

    if args.verify:
        print(f"Verifying benchmark data in {args.output_dir}...")
        results = verify_download(args.output_dir)

        print("\nVerification Results:")
        print(f"  Valid: {results['valid']}")
        print(f"  Trial files found: {results['trials_found']}")
        print(f"  Audio files found: {results['audio_found']}")

        if results["missing_trials"]:
            print(f"  Missing trials: {results['missing_trials']}")
        if results["missing_audio_dirs"]:
            print(f"  Missing audio dirs: {results['missing_audio_dirs']}")

        if results["valid"]:
            print("\n✓ Benchmark data is valid and ready to use.")
        else:
            print("\n✗ Benchmark data has issues. Try re-downloading.")
    else:
        download_benchmark(
            output_dir=args.output_dir,
            audio_only=args.audio_only,
            token=args.token,
        )


if __name__ == "__main__":
    main()
