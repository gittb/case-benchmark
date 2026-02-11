#!/usr/bin/env python3
"""Generate leaderboard.json from all model results in results/ directory.

Usage:
    python scripts/generate_leaderboard.py
    python scripts/generate_leaderboard.py --results-dir custom_results/ --output leaderboard.json
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load all result files from a directory."""
    results = []

    for result_file in results_dir.glob("**/results.json"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Extract key metrics
            entry = {
                "model_name": data.get("model_name", result_file.parent.name),
                "case_score": data.get("case_score"),
                "clean_eer": None,
                "codec_eer": None,
                "mic_eer": None,
                "noise_eer": None,
                "reverb_eer": None,
                "playback_eer": None,
                "result_file": str(result_file),
            }

            # Extract category EERs
            category_summary = data.get("category_summary", {})
            for cat in ["clean", "codec", "mic", "noise", "reverb", "playback"]:
                if cat in category_summary:
                    entry[f"{cat}_eer"] = category_summary[cat].get("avg_eer")

            results.append(entry)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue

    return results


def generate_leaderboard(results: list[dict]) -> dict:
    """Generate leaderboard from results."""
    # Sort by CASE-Score (lower is better)
    sorted_results = sorted(
        results,
        key=lambda x: x["case_score"] if x["case_score"] is not None else float("inf"),
    )

    # Add rank
    for i, entry in enumerate(sorted_results, 1):
        entry["rank"] = i

    # Format for output
    leaderboard = {
        "version": "1.0.0",
        "last_updated": __import__("datetime").datetime.now().isoformat(),
        "entries": sorted_results,
        "metrics": {
            "primary": "case_score",
            "secondary": ["clean_eer", "playback_eer"],
            "categories": ["clean", "codec", "mic", "noise", "reverb", "playback"],
        },
    }

    return leaderboard


def print_leaderboard(leaderboard: dict) -> None:
    """Print leaderboard in readable format."""
    print("\n" + "=" * 80)
    print("CASE Benchmark Leaderboard")
    print("=" * 80)
    print()

    header = f"{'Rank':<6} {'Model':<30} {'CASE-Score':<12} {'Clean':<8} {'Playback':<10}"
    print(header)
    print("-" * 80)

    for entry in leaderboard["entries"]:
        clean = entry.get("clean_eer")
        playback = entry.get("playback_eer")

        clean_str = f"{clean*100:.2f}%" if clean else "N/A"
        playback_str = f"{playback*100:.2f}%" if playback else "N/A"

        row = f"{entry['rank']:<6} {entry['model_name']:<30} {entry['case_score']:<12.3f} {clean_str:<8} {playback_str:<10}"
        print(row)

    print("-" * 80)
    print(f"\nTotal entries: {len(leaderboard['entries'])}")
    print(f"Last updated: {leaderboard['last_updated']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate CASE Benchmark leaderboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing model results (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/leaderboard.json"),
        help="Output leaderboard file (default: results/leaderboard.json)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_leaderboard",
        help="Print leaderboard to console",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found. Run evaluations first.")
        return

    print(f"Found {len(results)} result files.")

    # Generate leaderboard
    leaderboard = generate_leaderboard(results)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Leaderboard saved to {args.output}")

    # Print if requested
    if args.print_leaderboard:
        print_leaderboard(leaderboard)


if __name__ == "__main__":
    main()
