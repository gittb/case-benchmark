"""Evaluation metrics for speaker verification.

This module implements the standard metrics used in speaker verification:
- EER (Equal Error Rate): The point where FAR = FRR
- minDCF (minimum Detection Cost Function): NIST SRE standard metric
- CASE-Score: Weighted average of normalized EERs measuring carrier robustness

The CASE-Score formula:
    CASE-Score = Σ(weight_i × EER_i / EER_clean) / Σ(weight_i)

Where:
    - EER_i = EER on degraded protocol i (codec, reverb, noise, mic, playback)
    - EER_clean = EER on clean audio (baseline for normalization)
    - CASE-Score = 1.0 means no degradation (ideal)
    - CASE-Score > 1.0 means performance degrades on carrier-affected audio
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER).

    EER is the operating point where False Acceptance Rate equals False Rejection Rate.
    It's a standard metric for speaker verification that doesn't require choosing a threshold.

    Args:
        scores: Similarity scores (higher = more similar). Shape: (N,)
        labels: Binary labels (1 = same speaker, 0 = different speaker). Shape: (N,)

    Returns:
        Tuple of (EER as fraction between 0 and 1, decision threshold at EER)

    Example:
        >>> scores = np.array([0.9, 0.8, 0.3, 0.2])
        >>> labels = np.array([1, 1, 0, 0])
        >>> eer, threshold = compute_eer(scores, labels)
        >>> print(f"EER: {eer*100:.2f}%")
    """
    # Get ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER: the point where FPR = FNR
    try:
        # Use Brent's method for accurate EER via interpolation
        diff = fpr - fnr
        idx = np.argmin(np.abs(diff))

        if idx > 0 and idx < len(fpr) - 1:
            # Interpolate to find exact crossing point
            eer = brentq(
                lambda x: interp1d(fpr, fnr)(x) - x,
                fpr[max(0, idx - 1)],
                fpr[min(len(fpr) - 1, idx + 1)],
            )
        else:
            eer = fpr[idx]

        threshold = thresholds[idx]
    except (ValueError, IndexError):
        # Fallback: use closest point where FPR ≈ FNR
        idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    """Compute minimum Detection Cost Function (minDCF).

    minDCF is the standard metric used in NIST Speaker Recognition Evaluations.
    It weights false accepts and false rejects according to prior probabilities
    and costs, then finds the minimum achievable cost.

    The normalized minDCF is computed as:
        DCF = C_miss × P_miss × P_target + C_fa × P_fa × (1 - P_target)
        minDCF = min(DCF) / min(C_miss × P_target, C_fa × (1 - P_target))

    Args:
        scores: Similarity scores (higher = more similar). Shape: (N,)
        labels: Binary labels (1 = same speaker, 0 = different speaker). Shape: (N,)
        p_target: Prior probability of target speaker (default: 0.01 for SRE)
        c_miss: Cost of missed detection / false rejection (default: 1)
        c_fa: Cost of false alarm / false acceptance (default: 1)

    Returns:
        Tuple of (normalized minDCF, decision threshold at minimum DCF)

    Example:
        >>> scores = np.array([0.9, 0.8, 0.3, 0.2])
        >>> labels = np.array([1, 1, 0, 0])
        >>> min_dcf, threshold = compute_min_dcf(scores, labels)
        >>> print(f"minDCF: {min_dcf:.4f}")
    """
    # Get ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Compute DCF for all thresholds
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

    # Find minimum DCF
    min_idx = np.argmin(dcf)
    min_dcf = dcf[min_idx]
    threshold = thresholds[min_idx]

    # Normalize by the prior-weighted minimum cost
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf_norm = min_dcf / c_default

    return float(min_dcf_norm), float(threshold)


def compute_far_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Compute FAR and FRR at a given threshold.

    Args:
        scores: Similarity scores (higher = more similar). Shape: (N,)
        labels: Binary labels (1 = same speaker, 0 = different speaker). Shape: (N,)
        threshold: Decision threshold (accept if score >= threshold)

    Returns:
        Tuple of (False Acceptance Rate, False Rejection Rate)
    """
    predictions = (scores >= threshold).astype(int)

    # False Acceptance Rate: impostors incorrectly accepted
    neg_mask = labels == 0
    far = np.sum(predictions[neg_mask]) / np.sum(neg_mask) if np.sum(neg_mask) > 0 else 0.0

    # False Rejection Rate: targets incorrectly rejected
    pos_mask = labels == 1
    frr = 1 - np.sum(predictions[pos_mask]) / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0

    return float(far), float(frr)


def compute_cosine_scores(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between pairs of embeddings.

    Args:
        embeddings1: First set of embeddings. Shape: (N, D)
        embeddings2: Second set of embeddings. Shape: (N, D)

    Returns:
        Cosine similarity scores in range [-1, 1]. Shape: (N,)
    """
    # L2 normalize
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Avoid division by zero
    norm1 = np.maximum(norm1, 1e-8)
    norm2 = np.maximum(norm2, 1e-8)

    embeddings1_norm = embeddings1 / norm1
    embeddings2_norm = embeddings2 / norm2

    # Compute cosine similarity (dot product of normalized vectors)
    scores = np.sum(embeddings1_norm * embeddings2_norm, axis=1)

    return scores


@dataclass
class CASEScoreResult:
    """Result of CASE-Score computation."""

    case_score: float
    clean_eer: float
    category_results: dict[str, dict]
    # category_results structure:
    # {
    #     "codec": {"avg_eer": 0.02, "normalized": 2.0, "n_protocols": 7},
    #     "playback": {"avg_eer": 0.15, "normalized": 15.0, "n_protocols": 3},
    #     ...
    # }


@dataclass
class CASEScoreV2Result:
    """Result of CASE-Score v2 computation with dual metrics.

    Provides two complementary metrics:
    1. case_score_absolute: Weighted average of EERs (lower = better performance)
    2. degradation_factor: Weighted average of (EER_degraded - EER_clean) (lower = more robust)
    """

    case_score_absolute: float  # Direct performance measure (lower is better)
    degradation_factor: float   # Robustness measure (lower is better)
    clean_eer: float
    category_results: dict[str, dict]
    # category_results structure:
    # {
    #     "codec": {
    #         "avg_eer": 0.02,           # Average EER in this category
    #         "degradation": 0.01,       # avg_eer - clean_eer
    #         "weight": 1.0,
    #         "n_protocols": 7
    #     },
    #     ...
    # }


# Default weights for CASE-Score computation
DEFAULT_WEIGHTS = {
    "clean": 1.0,
    "codec": 1.0,
    "reverb": 1.0,
    "noise": 1.0,
    "mic": 1.0,
    "playback": 1.0,
}


def compute_case_score(
    protocol_results: dict[str, dict],
    weights: dict[str, float] | None = None,
    normalize_to_clean: bool = True,
) -> CASEScoreResult:
    """Compute the CASE-Score from protocol evaluation results.

    The CASE-Score measures how much a model's performance degrades on
    carrier-affected audio compared to clean audio. Lower is better.

    Formula:
        CASE-Score = Σ(weight_i × EER_i / EER_clean) / Σ(weight_i)

    A CASE-Score of 1.0 means no degradation (ideal).
    A CASE-Score of 5.0 means average 5× worse performance on degraded audio.

    Args:
        protocol_results: Dictionary mapping protocol name to results dict.
            Each results dict must contain 'eer' key with EER as a fraction.
            Example: {"clean_clean": {"eer": 0.01}, "clean_codec_gsm": {"eer": 0.03}}
        weights: Optional weights for each protocol category.
            Default: equal weights for all categories.
        normalize_to_clean: If True, normalize EERs to clean baseline.
            If False, return raw weighted average of EERs.

    Returns:
        CASEScoreResult with the CASE-Score and category breakdown.

    Example:
        >>> results = {
        ...     "clean_clean": {"eer": 0.01},
        ...     "clean_codec_gsm": {"eer": 0.025},
        ...     "clean_codec_opus": {"eer": 0.015},
        ... }
        >>> score = compute_case_score(results)
        >>> print(f"CASE-Score: {score.case_score:.2f}")
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Find clean EER for normalization
    clean_eer = None
    for name, res in protocol_results.items():
        if name == "clean_clean" or name.endswith("_clean"):
            eer = res.get("eer")
            if eer is not None and not np.isnan(eer):
                clean_eer = eer
                break

    if clean_eer is None or clean_eer == 0:
        clean_eer = 0.01  # Default to 1% to avoid division by zero

    # Group results by protocol category
    protocol_eers: dict[str, list[float]] = defaultdict(list)

    for name, res in protocol_results.items():
        eer = res.get("eer")
        if eer is None or np.isnan(eer):
            continue

        # Extract category from protocol name
        # e.g., "clean_codec_gsm" -> "codec", "clean_reverb" -> "reverb"
        parts = name.split("_")
        if len(parts) >= 2:
            category = parts[1]  # Second part is usually the category
        else:
            category = parts[0]

        protocol_eers[category].append(eer)

    # Compute weighted average of normalized EERs
    category_results = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for category, eers in protocol_eers.items():
        if not eers:
            continue

        avg_eer = float(np.mean(eers))
        weight = weights.get(category, 1.0)

        if normalize_to_clean and category != "clean":
            normalized_eer = avg_eer / clean_eer
        else:
            normalized_eer = avg_eer

        category_results[category] = {
            "avg_eer": avg_eer,
            "normalized": normalized_eer,
            "weight": weight,
            "n_protocols": len(eers),
        }

        weighted_sum += weight * normalized_eer
        total_weight += weight

    if total_weight > 0:
        case_score = weighted_sum / total_weight
    else:
        case_score = float("nan")

    return CASEScoreResult(
        case_score=case_score,
        clean_eer=clean_eer,
        category_results=category_results,
    )


def compute_case_score_v2(
    protocol_results: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> CASEScoreV2Result:
    """Compute the CASE-Score v2 with dual metrics.

    This version addresses the normalization issue in v1 by providing two
    complementary metrics:

    1. **CASE-Score (Absolute)**: Weighted average of EERs across categories
       - Lower is better
       - Measures actual speaker verification performance
       - Good for ranking models by overall capability

    2. **Degradation Factor**: Weighted average of (EER_degraded - EER_clean)
       - Lower is better (0 = no degradation)
       - Measures how much performance is lost due to carrier mismatch
       - Good for measuring robustness independent of baseline performance

    The v1 formula (EER_degraded / EER_clean) penalized models with good clean
    performance. For example:
      - Model A: Clean 5% → Degraded 10% = Score 2.0
      - Model B: Clean 0.5% → Degraded 10% = Score 20.0

    Despite Model B being better (lower degraded EER), it got a worse score.
    V2 fixes this by measuring absolute degradation instead.

    Args:
        protocol_results: Dictionary mapping protocol name to results dict.
            Each results dict must contain 'eer' key with EER as a fraction.
        weights: Optional weights for each protocol category.
            Default: equal weights for all categories.

    Returns:
        CASEScoreV2Result with both absolute and degradation metrics.

    Example:
        >>> results = {
        ...     "clean_clean": {"eer": 0.005},
        ...     "clean_codec_gsm": {"eer": 0.025},
        ...     "clean_reverb": {"eer": 0.05},
        ... }
        >>> score = compute_case_score_v2(results)
        >>> print(f"CASE-Score: {score.case_score_absolute*100:.2f}%")
        >>> print(f"Degradation: {score.degradation_factor*100:.2f}%")
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Find clean EER for degradation calculation
    clean_eer = None
    for name, res in protocol_results.items():
        if name == "clean_clean" or name.endswith("_clean"):
            eer = res.get("eer")
            if eer is not None and not np.isnan(eer):
                clean_eer = eer
                break

    if clean_eer is None or np.isnan(clean_eer):
        clean_eer = 0.01  # Default to 1% if not found

    # Group results by protocol category
    protocol_eers: dict[str, list[float]] = defaultdict(list)

    for name, res in protocol_results.items():
        eer = res.get("eer")
        if eer is None or np.isnan(eer):
            continue

        # Extract category from protocol name
        # e.g., "clean_codec_gsm" -> "codec", "clean_reverb" -> "reverb"
        parts = name.split("_")
        if len(parts) >= 2:
            category = parts[1]  # Second part is usually the category
        else:
            category = parts[0]

        protocol_eers[category].append(eer)

    # Compute both metrics
    category_results = {}
    total_weight = 0.0
    weighted_eer_sum = 0.0
    weighted_degradation_sum = 0.0

    for category, eers in protocol_eers.items():
        if not eers:
            continue

        avg_eer = float(np.mean(eers))
        weight = weights.get(category, 1.0)

        # Degradation is how much worse than clean (0 for clean category)
        if category == "clean":
            degradation = 0.0
        else:
            degradation = avg_eer - clean_eer

        category_results[category] = {
            "avg_eer": avg_eer,
            "degradation": degradation,
            "weight": weight,
            "n_protocols": len(eers),
        }

        weighted_eer_sum += weight * avg_eer
        weighted_degradation_sum += weight * degradation
        total_weight += weight

    if total_weight > 0:
        case_score_absolute = weighted_eer_sum / total_weight
        degradation_factor = weighted_degradation_sum / total_weight
    else:
        case_score_absolute = float("nan")
        degradation_factor = float("nan")

    return CASEScoreV2Result(
        case_score_absolute=case_score_absolute,
        degradation_factor=degradation_factor,
        clean_eer=clean_eer,
        category_results=category_results,
    )
