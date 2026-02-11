"""Main evaluation runner for the CASE Benchmark.

This module provides the CASEBenchmark class for evaluating speaker embedding models
across all CASE Benchmark protocols.

Usage:
    from case_benchmark import CASEBenchmark, load_model

    # Load benchmark and model
    benchmark = CASEBenchmark("/path/to/benchmark")
    model = load_model("speechbrain")

    # Run full evaluation
    results = benchmark.evaluate(model)

    # Display results
    results.print_summary()
    results.save("results/my_model.json")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from case_benchmark.metrics import (
    compute_cosine_scores,
    compute_eer,
    compute_min_dcf,
    compute_case_score,
    compute_case_score_v2,
    CASEScoreResult,
    CASEScoreV2Result,
)
from case_benchmark.utils import load_audio, load_trials, get_protocol_category

if TYPE_CHECKING:
    from case_benchmark.models.base import EmbeddingModel


@dataclass
class ProtocolResult:
    """Results for a single evaluation protocol."""

    name: str
    eer: float
    eer_threshold: float
    min_dcf: float
    min_dcf_threshold: float
    num_trials: int
    num_targets: int
    num_impostors: int


@dataclass
class BenchmarkResults:
    """Complete results from CASE Benchmark evaluation."""

    model_name: str
    case_score: float
    case_score_details: CASEScoreResult
    protocol_results: dict[str, ProtocolResult] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    # V2 metrics (optional for backward compatibility)
    case_score_v2: CASEScoreV2Result | None = None

    def get_category_summary(self) -> dict[str, dict]:
        """Get summary statistics grouped by protocol category."""
        return self.case_score_details.category_results

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"CASE Benchmark Results: {self.model_name}")
        print(f"{'='*60}\n")

        # V2 metrics (if available)
        if self.case_score_v2:
            print("CASE-Score v2 Metrics:")
            print(f"  CASE-Score (Absolute): {self.case_score_v2.case_score_absolute*100:.2f}%")
            print(f"  Degradation Factor:    {self.case_score_v2.degradation_factor*100:.2f}%")
            print(f"  Clean EER:             {self.case_score_v2.clean_eer*100:.2f}%")
            print()

        # Legacy CASE-Score
        print(f"CASE-Score (v1): {self.case_score:.3f}")
        print(f"(Normalized ratio - 1.0 = no degradation from clean audio)\n")

        # Category breakdown with v2 metrics
        print("Category Breakdown:")
        print("-" * 60)
        if self.case_score_v2:
            print(f"{'Category':<15} {'Avg EER':>10} {'Degradation':>12} {'Protocols':>10}")
            print("-" * 60)
            for cat, data in sorted(self.case_score_v2.category_results.items()):
                eer_pct = data["avg_eer"] * 100
                deg_pct = data["degradation"] * 100
                n = data["n_protocols"]
                print(f"{cat:<15} {eer_pct:>9.2f}% {deg_pct:>+10.2f}% {n:>10}")
        else:
            print(f"{'Category':<15} {'Avg EER':>10} {'vs Clean':>12} {'Protocols':>10}")
            print("-" * 60)
            for cat, data in sorted(self.case_score_details.category_results.items()):
                eer_pct = data["avg_eer"] * 100
                norm = data["normalized"]
                n = data["n_protocols"]
                print(f"{cat:<15} {eer_pct:>9.2f}% {norm:>10.1f}× {n:>10}")

        print("-" * 60)

        # Per-protocol results
        print("\nPer-Protocol Results:")
        print("-" * 60)
        print(f"{'Protocol':<40} {'EER':>8} {'minDCF':>8} {'Trials':>8}")
        print("-" * 60)

        for name, res in sorted(self.protocol_results.items()):
            eer_pct = res.eer * 100
            print(f"{name:<40} {eer_pct:>7.2f}% {res.min_dcf:>8.4f} {res.num_trials:>8}")

        print("-" * 60)
        print()

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "model_name": self.model_name,
            "case_score": self.case_score,
            "config": self.config,
            "category_summary": {
                cat: {
                    "avg_eer": data["avg_eer"],
                    "normalized": data["normalized"],
                    "n_protocols": data["n_protocols"],
                }
                for cat, data in self.case_score_details.category_results.items()
            },
            "protocol_results": {
                name: {
                    "eer": res.eer,
                    "eer_threshold": res.eer_threshold,
                    "min_dcf": res.min_dcf,
                    "min_dcf_threshold": res.min_dcf_threshold,
                    "num_trials": res.num_trials,
                    "num_targets": res.num_targets,
                    "num_impostors": res.num_impostors,
                }
                for name, res in self.protocol_results.items()
            },
        }

        # Add v2 metrics if available
        if self.case_score_v2:
            output["case_score_v2"] = {
                "case_score_absolute": self.case_score_v2.case_score_absolute,
                "degradation_factor": self.case_score_v2.degradation_factor,
                "clean_eer": self.case_score_v2.clean_eer,
                "category_summary": {
                    cat: {
                        "avg_eer": data["avg_eer"],
                        "degradation": data["degradation"],
                        "weight": data["weight"],
                        "n_protocols": data["n_protocols"],
                    }
                    for cat, data in self.case_score_v2.category_results.items()
                },
            }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct CASEScoreResult
        case_score_details = CASEScoreResult(
            case_score=data["case_score"],
            clean_eer=data["category_summary"].get("clean", {}).get("avg_eer", 0.01),
            category_results=data["category_summary"],
        )

        # Reconstruct protocol results
        protocol_results = {}
        for name, res_data in data.get("protocol_results", {}).items():
            protocol_results[name] = ProtocolResult(
                name=name,
                eer=res_data["eer"],
                eer_threshold=res_data.get("eer_threshold", 0),
                min_dcf=res_data["min_dcf"],
                min_dcf_threshold=res_data.get("min_dcf_threshold", 0),
                num_trials=res_data["num_trials"],
                num_targets=res_data.get("num_targets", 0),
                num_impostors=res_data.get("num_impostors", 0),
            )

        # Reconstruct v2 metrics if present
        case_score_v2 = None
        if "case_score_v2" in data:
            v2_data = data["case_score_v2"]
            case_score_v2 = CASEScoreV2Result(
                case_score_absolute=v2_data["case_score_absolute"],
                degradation_factor=v2_data["degradation_factor"],
                clean_eer=v2_data["clean_eer"],
                category_results=v2_data["category_summary"],
            )

        return cls(
            model_name=data["model_name"],
            case_score=data["case_score"],
            case_score_details=case_score_details,
            protocol_results=protocol_results,
            config=data.get("config", {}),
            case_score_v2=case_score_v2,
        )


class CASEBenchmark:
    """CASE Benchmark evaluator.

    Evaluates speaker embedding models across all CASE Benchmark protocols,
    computing EER, minDCF, and the overall CASE-Score.

    Example:
        >>> benchmark = CASEBenchmark("/path/to/benchmark")
        >>> model = load_model("speechbrain")
        >>> results = benchmark.evaluate(model)
        >>> print(f"CASE-Score: {results.case_score:.2f}")
    """

    def __init__(
        self,
        benchmark_dir: str | Path,
        device: str = "cpu",
        cache_embeddings: bool = True,
    ):
        """Initialize CASE Benchmark evaluator.

        Args:
            benchmark_dir: Path to benchmark directory containing audio and trials
            device: Device to use for model inference ('cpu' or 'cuda')
            cache_embeddings: Whether to cache embeddings to avoid re-extraction
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.device = device
        self.cache_embeddings = cache_embeddings

        # Find trials directory
        self.trials_dir = self._find_trials_dir()

        # Embedding cache: path -> embedding
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _find_trials_dir(self) -> Path:
        """Find the trials directory within the benchmark."""
        candidates = [
            self.benchmark_dir / "trials",
            self.benchmark_dir / "benchmark" / "trials",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Trials directory not found in {self.benchmark_dir}. "
            "Expected 'trials/' or 'benchmark/trials/' subdirectory."
        )

    def list_protocols(self) -> list[str]:
        """List all available evaluation protocols."""
        trial_files = sorted(self.trials_dir.glob("*.txt"))
        return [f.stem for f in trial_files]

    def evaluate(
        self,
        model: "EmbeddingModel",
        protocols: list[str] | None = None,
        save_scores: bool = False,
        output_dir: str | Path | None = None,
        show_progress: bool = True,
    ) -> BenchmarkResults:
        """Run full CASE Benchmark evaluation.

        Args:
            model: Speaker embedding model to evaluate
            protocols: List of protocols to evaluate (default: all)
            save_scores: Whether to save per-trial scores
            output_dir: Directory to save scores (if save_scores=True)
            show_progress: Whether to show progress bars

        Returns:
            BenchmarkResults with CASE-Score and per-protocol metrics
        """
        # Ensure model is loaded
        if not model.is_loaded:
            model.load(self.device)

        # Get protocols to evaluate
        available = self.list_protocols()
        if protocols is None:
            protocols = available
        else:
            # Validate protocols
            invalid = set(protocols) - set(available)
            if invalid:
                raise ValueError(f"Unknown protocols: {invalid}")

        print(f"\nEvaluating {model.name} on CASE Benchmark")
        print(f"Protocols: {len(protocols)}")
        print(f"Device: {self.device}")
        print()

        # Evaluate each protocol
        all_results: dict[str, dict] = {}

        for protocol in tqdm(protocols, desc="Protocols", disable=not show_progress):
            result = self._evaluate_protocol(
                model=model,
                protocol=protocol,
                show_progress=show_progress,
            )

            if result is not None:
                all_results[protocol] = {
                    "eer": result.eer,
                    "min_dcf": result.min_dcf,
                    "num_trials": result.num_trials,
                }

                if save_scores and output_dir:
                    # Save scores for this protocol
                    pass  # TODO: implement score saving

        # Compute CASE-Score (v1 for backward compatibility)
        case_score_result = compute_case_score(all_results, normalize_to_clean=True)

        # Compute CASE-Score v2 (new dual metrics)
        case_score_v2_result = compute_case_score_v2(all_results)

        # Build protocol results
        protocol_results = {}
        for protocol, res_dict in all_results.items():
            protocol_results[protocol] = ProtocolResult(
                name=protocol,
                eer=res_dict.get("eer", float("nan")),
                eer_threshold=res_dict.get("eer_threshold", 0),
                min_dcf=res_dict.get("min_dcf", float("nan")),
                min_dcf_threshold=res_dict.get("min_dcf_threshold", 0),
                num_trials=res_dict.get("num_trials", 0),
                num_targets=res_dict.get("num_targets", 0),
                num_impostors=res_dict.get("num_impostors", 0),
            )

        return BenchmarkResults(
            model_name=model.name,
            case_score=case_score_result.case_score,
            case_score_details=case_score_result,
            protocol_results=protocol_results,
            config={
                "benchmark_dir": str(self.benchmark_dir),
                "device": self.device,
            },
            case_score_v2=case_score_v2_result,
        )

    def _evaluate_protocol(
        self,
        model: "EmbeddingModel",
        protocol: str,
        show_progress: bool = True,
    ) -> ProtocolResult | None:
        """Evaluate a single protocol."""
        trials_path = self.trials_dir / f"{protocol}.txt"

        if not trials_path.exists():
            print(f"Warning: Trial file not found: {trials_path}")
            return None

        # Load trials
        enroll_paths, test_paths, labels = load_trials(trials_path)

        if len(enroll_paths) == 0:
            return None

        # Extract embeddings and compute scores
        scores = []
        valid_labels = []

        iterator = zip(enroll_paths, test_paths, labels)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc=f"  {protocol}",
                leave=False,
            )

        for enroll_path, test_path, label in iterator:
            # Resolve paths relative to benchmark directory
            enroll_full = self.benchmark_dir / enroll_path
            test_full = self.benchmark_dir / test_path

            if not enroll_full.exists() or not test_full.exists():
                continue

            try:
                # Get embeddings (with caching)
                emb1 = self._get_embedding(model, enroll_full)
                emb2 = self._get_embedding(model, test_full)

                # Compute cosine similarity
                score = compute_cosine_scores(
                    emb1.reshape(1, -1),
                    emb2.reshape(1, -1),
                )[0]

                scores.append(score)
                valid_labels.append(label)
            except Exception as e:
                # Skip problematic files
                continue

        if len(scores) == 0:
            return None

        scores_array = np.array(scores)
        labels_array = np.array(valid_labels)

        # Compute metrics
        eer, eer_threshold = compute_eer(scores_array, labels_array)
        min_dcf, dcf_threshold = compute_min_dcf(scores_array, labels_array)

        return ProtocolResult(
            name=protocol,
            eer=eer,
            eer_threshold=eer_threshold,
            min_dcf=min_dcf,
            min_dcf_threshold=dcf_threshold,
            num_trials=len(scores),
            num_targets=int(np.sum(labels_array)),
            num_impostors=int(len(labels_array) - np.sum(labels_array)),
        )

    def _get_embedding(self, model: "EmbeddingModel", audio_path: Path) -> np.ndarray:
        """Get embedding for an audio file, using cache if available."""
        path_str = str(audio_path)

        if self.cache_embeddings and path_str in self._embedding_cache:
            return self._embedding_cache[path_str]

        embedding = model.extract_embedding(audio_path)

        if self.cache_embeddings:
            self._embedding_cache[path_str] = embedding

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
