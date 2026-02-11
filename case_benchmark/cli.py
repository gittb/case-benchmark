"""Command-line interface for CASE Benchmark.

Usage:
    case-benchmark evaluate --model speechbrain --benchmark-dir /path/to/benchmark
    case-benchmark list-models
    case-benchmark download --output-dir /path/to/download

For more information:
    case-benchmark --help
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for case-benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="case-benchmark",
        description="CASE Benchmark: Carrier-Agnostic Speaker Verification Evaluation",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model on the CASE Benchmark",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., 'speechbrain', 'wespeaker_resnet34')",
    )
    eval_parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Path to benchmark directory containing audio and trials",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run evaluation on (default: cpu)",
    )
    eval_parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=None,
        help="Specific protocols to evaluate (default: all)",
    )
    eval_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding caching (slower but uses less memory)",
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available models",
    )

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download CASE Benchmark data from HuggingFace",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_data"),
        help="Output directory for downloaded data",
    )
    download_parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download only audio files (skip trial lists)",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "list-models":
        run_list_models()
    elif args.command == "download":
        run_download(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_evaluate(args):
    """Run benchmark evaluation."""
    from case_benchmark import CASEBenchmark, load_model

    print(f"CASE Benchmark Evaluation")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Device: {args.device}")
    print()

    try:
        # Load model
        print(f"Loading model: {args.model}...")
        model = load_model(args.model, device=args.device)
        print(f"Model loaded: {model}")

        # Initialize benchmark
        benchmark = CASEBenchmark(
            benchmark_dir=args.benchmark_dir,
            device=args.device,
            cache_embeddings=not args.no_cache,
        )

        # List available protocols
        available_protocols = benchmark.list_protocols()
        print(f"Available protocols: {len(available_protocols)}")

        # Run evaluation
        results = benchmark.evaluate(
            model=model,
            protocols=args.protocols,
            show_progress=True,
        )

        # Display results
        results.print_summary()

        # Save results
        output_path = args.output_dir / f"{args.model.replace('/', '_')}_results.json"
        results.save(output_path)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_list_models():
    """List available models."""
    from case_benchmark.models import list_available_models

    print("Available Models")
    print("=" * 50)

    models = list_available_models()
    if not models:
        print("No models registered.")
        print("\nTo use models, install the required dependencies:")
        print("  pip install case-benchmark[speechbrain]")
        print("  pip install case-benchmark[wespeaker]")
        print("  pip install case-benchmark[pyannote]")
        print("  pip install case-benchmark[nemo]")
        print("  pip install case-benchmark[resemblyzer]")
        print("  pip install case-benchmark[all-models]")
    else:
        for model_name in models:
            print(f"  - {model_name}")

    print()


def run_download(args):
    """Download benchmark data."""
    print("Download CASE Benchmark Data")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print()

    try:
        from case_benchmark.download import download_benchmark

        download_benchmark(
            output_dir=args.output_dir,
            audio_only=args.audio_only,
        )
    except ImportError:
        print("Download functionality requires huggingface_hub.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
