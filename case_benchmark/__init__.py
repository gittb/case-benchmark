"""CASE Benchmark: Carrier-Agnostic Speaker Verification Evaluation.

This package provides tools to evaluate speaker embedding models on the CASE Benchmark,
which measures robustness across real-world acoustic carrier conditions.

Quick Start:
    >>> from case_benchmark import CASEBenchmark, load_model
    >>> benchmark = CASEBenchmark("/path/to/benchmark")
    >>> model = load_model("speechbrain")
    >>> results = benchmark.evaluate(model)
    >>> print(f"CASE-Score: {results.case_score:.2f}")
"""

__version__ = "1.0.0"

from case_benchmark.evaluate import CASEBenchmark, BenchmarkResults
from case_benchmark.metrics import compute_eer, compute_min_dcf, compute_case_score
from case_benchmark.models import load_model, list_available_models

__all__ = [
    "CASEBenchmark",
    "BenchmarkResults",
    "compute_eer",
    "compute_min_dcf",
    "compute_case_score",
    "load_model",
    "list_available_models",
    "__version__",
]
