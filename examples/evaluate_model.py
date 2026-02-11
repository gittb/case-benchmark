#!/usr/bin/env python3
"""Example: Evaluate a speaker embedding model on the CASE Benchmark.

This example shows how to:
1. Load the CASE Benchmark
2. Load a pretrained speaker embedding model
3. Run evaluation on all protocols
4. Display and save results

Usage:
    python examples/evaluate_model.py --model speechbrain --benchmark-dir /path/to/benchmark

Requirements:
    pip install case-benchmark[speechbrain]
"""

import argparse
from pathlib import Path

from case_benchmark import CASEBenchmark, load_model, list_available_models


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the CASE Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="speechbrain",
        help="Model to evaluate (see --list-models)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Path to CASE Benchmark directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Available models:")
        for model_name in list_available_models():
            print(f"  - {model_name}")
        print("\nInstall model dependencies:")
        print("  pip install case-benchmark[speechbrain]")
        print("  pip install case-benchmark[wespeaker]")
        print("  pip install case-benchmark[all-models]")
        return

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = load_model(args.model, device=args.device)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nRun with --list-models to see available models.")
        return
    except ImportError as e:
        print(f"Error: {e}")
        print(f"\nInstall with: pip install case-benchmark[{args.model}]")
        return

    print(f"Model loaded: {model}")
    print(f"  Embedding dimension: {model.embedding_dim}")
    print()

    # Initialize benchmark
    print(f"Loading benchmark from: {args.benchmark_dir}")
    benchmark = CASEBenchmark(
        benchmark_dir=args.benchmark_dir,
        device=args.device,
    )

    # List available protocols
    protocols = benchmark.list_protocols()
    print(f"Found {len(protocols)} evaluation protocols")
    print()

    # Run evaluation
    print("Starting evaluation...")
    print("=" * 60)

    results = benchmark.evaluate(
        model=model,
        show_progress=True,
    )

    # Display results
    results.print_summary()

    # Save results
    output_path = args.output_dir / f"{args.model.replace('/', '_')}_results.json"
    results.save(output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
