#!/usr/bin/env python3
"""Upload CASE Benchmark to HuggingFace with rate limit handling.

This script uploads the benchmark in smaller batches to avoid hitting
the 1,000 requests per 5 minutes rate limit on HuggingFace free tier.
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files


def get_already_uploaded(api: HfApi, repo_id: str) -> set[str]:
    """Get list of files already uploaded to the repo."""
    try:
        files = set(api.list_repo_files(repo_id, repo_type="dataset"))
        return files
    except Exception:
        return set()


def upload_folder_batch(
    api: HfApi,
    repo_id: str,
    local_folder: Path,
    path_in_repo: str,
    already_uploaded: set[str],
    dry_run: bool = False,
) -> int:
    """Upload a folder, skipping already uploaded files.

    Returns number of new files uploaded.
    """
    # Count files that would be uploaded
    all_files = list(local_folder.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    new_files = []
    for f in all_files:
        repo_path = f"{path_in_repo}/{f.relative_to(local_folder)}"
        if repo_path not in already_uploaded:
            new_files.append(f)

    if not new_files:
        print(f"  All {len(all_files)} files already uploaded, skipping")
        return 0

    print(f"  Uploading {len(new_files)} new files (skipping {len(all_files) - len(new_files)} existing)")

    if dry_run:
        return len(new_files)

    api.upload_folder(
        folder_path=str(local_folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )

    return len(new_files)


def main():
    parser = argparse.ArgumentParser(description="Upload CASE Benchmark to HuggingFace")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/hdd_nas/datasets/case/benchmark"),
        help="Path to benchmark directory",
    )
    parser.add_argument(
        "--repo-id",
        default="bigstorm/case-benchmark",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=300,
        help="Seconds to wait between batches (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--dataset",
        choices=["voxceleb1_o", "librispeech", "all"],
        default="all",
        help="Which dataset to upload",
    )
    args = parser.parse_args()

    api = HfApi()
    benchmark_dir = args.benchmark_dir

    print(f"Checking existing files in {args.repo_id}...")
    already_uploaded = get_already_uploaded(api, args.repo_id)
    print(f"Found {len(already_uploaded)} files already uploaded\n")

    # Define upload batches - each condition is a separate batch
    datasets = ["voxceleb1_o", "librispeech"] if args.dataset == "all" else [args.dataset]
    conditions = ["clean", "codec", "mic", "noise", "reverb", "playback"]

    batches = []
    for dataset in datasets:
        for condition in conditions:
            local_path = benchmark_dir / dataset / condition
            if local_path.exists():
                batches.append({
                    "local": local_path,
                    "repo_path": f"{dataset}/{condition}",
                    "name": f"{dataset}/{condition}",
                })

    print(f"Upload plan: {len(batches)} batches")
    for b in batches:
        file_count = len(list(b["local"].rglob("*.wav")))
        print(f"  - {b['name']}: ~{file_count} files")
    print()

    total_uploaded = 0
    for i, batch in enumerate(batches):
        print(f"\n[{i+1}/{len(batches)}] Uploading {batch['name']}...")

        count = upload_folder_batch(
            api,
            args.repo_id,
            batch["local"],
            batch["repo_path"],
            already_uploaded,
            dry_run=args.dry_run,
        )
        total_uploaded += count

        # Wait between batches (except after the last one)
        if count > 0 and i < len(batches) - 1 and not args.dry_run:
            print(f"\n  Waiting {args.delay}s before next batch (rate limit cooldown)...")
            time.sleep(args.delay)

    print(f"\n\nDone! Total files uploaded: {total_uploaded}")
    if args.dry_run:
        print("(Dry run - no actual uploads performed)")


if __name__ == "__main__":
    main()
