"""Tests for CASE Benchmark metrics."""

import numpy as np
import pytest

from case_benchmark.metrics import (
    compute_eer,
    compute_min_dcf,
    compute_cosine_scores,
    compute_case_score,
    compute_case_score_v2,
)


class TestComputeEER:
    """Tests for compute_eer function."""

    def test_perfect_separation(self):
        """Test EER with perfectly separable scores."""
        # Targets all have score > 0.5, impostors all have score < 0.5
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        eer, threshold = compute_eer(scores, labels)

        assert eer == pytest.approx(0.0, abs=0.01)
        assert 0.4 <= threshold <= 0.6

    def test_random_scores(self):
        """Test EER with random scores (should be ~50%)."""
        np.random.seed(42)
        scores = np.random.rand(1000)
        labels = np.random.randint(0, 2, 1000)

        eer, threshold = compute_eer(scores, labels)

        # Random classifier should have ~50% EER
        assert 0.4 <= eer <= 0.6

    def test_realistic_scores(self):
        """Test EER with realistic speaker verification scores."""
        np.random.seed(42)
        # Targets: higher scores with some overlap
        target_scores = np.random.normal(0.7, 0.15, 500)
        # Impostors: lower scores with some overlap
        impostor_scores = np.random.normal(0.3, 0.15, 500)

        scores = np.concatenate([target_scores, impostor_scores])
        labels = np.concatenate([np.ones(500), np.zeros(500)])

        eer, threshold = compute_eer(scores, labels)

        # Should have reasonable EER given the overlap
        assert 0.0 <= eer <= 0.5
        assert 0.0 <= threshold <= 1.0


class TestComputeMinDCF:
    """Tests for compute_min_dcf function."""

    def test_perfect_separation(self):
        """Test minDCF with perfectly separable scores."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        min_dcf, threshold = compute_min_dcf(scores, labels)

        assert min_dcf == pytest.approx(0.0, abs=0.01)

    def test_random_scores(self):
        """Test minDCF with random scores."""
        np.random.seed(42)
        scores = np.random.rand(1000)
        labels = np.random.randint(0, 2, 1000)

        min_dcf, threshold = compute_min_dcf(scores, labels)

        # Random classifier should have minDCF close to 1
        assert 0.5 <= min_dcf <= 1.0


class TestComputeCosineScores:
    """Tests for compute_cosine_scores function."""

    def test_identical_embeddings(self):
        """Test cosine similarity of identical embeddings."""
        emb = np.array([[1.0, 0.0, 0.0]])
        scores = compute_cosine_scores(emb, emb)

        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_embeddings(self):
        """Test cosine similarity of orthogonal embeddings."""
        emb1 = np.array([[1.0, 0.0, 0.0]])
        emb2 = np.array([[0.0, 1.0, 0.0]])
        scores = compute_cosine_scores(emb1, emb2)

        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_embeddings(self):
        """Test cosine similarity of opposite embeddings."""
        emb1 = np.array([[1.0, 0.0, 0.0]])
        emb2 = np.array([[-1.0, 0.0, 0.0]])
        scores = compute_cosine_scores(emb1, emb2)

        assert scores[0] == pytest.approx(-1.0, abs=1e-6)

    def test_batch_computation(self):
        """Test batch computation of cosine scores."""
        emb1 = np.random.randn(100, 192)
        emb2 = np.random.randn(100, 192)
        scores = compute_cosine_scores(emb1, emb2)

        assert scores.shape == (100,)
        assert all(-1 <= s <= 1 for s in scores)


class TestComputeCASEScore:
    """Tests for compute_case_score function."""

    def test_no_degradation(self):
        """Test CASE-Score when all protocols match clean."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.01},
            "clean_codec_ulaw": {"eer": 0.01},
            "clean_reverb": {"eer": 0.01},
        }

        score_result = compute_case_score(results)

        # All protocols equal to clean = normalized ratio of 1.0 for degraded categories
        # The formula includes clean (not normalized) + normalized degraded categories
        # clean=0.01, codec=1.0 (0.01/0.01), reverb=1.0 (0.01/0.01)
        # weighted avg = (0.01 + 1.0 + 1.0) / 3 = 0.67
        assert score_result.case_score == pytest.approx(0.67, abs=0.1)

    def test_with_degradation(self):
        """Test CASE-Score with realistic degradation."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.03},  # 3x worse
            "clean_reverb": {"eer": 0.10},  # 10x worse
        }

        score_result = compute_case_score(results)

        # Should be > 1.0 due to degradation
        assert score_result.case_score > 1.0
        assert score_result.clean_eer == pytest.approx(0.01, abs=1e-6)

    def test_category_grouping(self):
        """Test that protocols are correctly grouped by category."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.02},
            "clean_codec_ulaw": {"eer": 0.025},
            "clean_codec_alaw": {"eer": 0.022},
        }

        score_result = compute_case_score(results)

        # Should have clean and codec categories
        assert "clean" in score_result.category_results
        assert "codec" in score_result.category_results

        # Codec should have 3 protocols
        assert score_result.category_results["codec"]["n_protocols"] == 3

        # Codec avg should be mean of the three
        expected_avg = (0.02 + 0.025 + 0.022) / 3
        assert score_result.category_results["codec"]["avg_eer"] == pytest.approx(
            expected_avg, abs=1e-6
        )


class TestComputeCASEScoreV2:
    """Tests for compute_case_score_v2 function."""

    def test_no_degradation(self):
        """Test V2 metrics when all protocols match clean."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.01},
            "clean_codec_ulaw": {"eer": 0.01},
            "clean_reverb": {"eer": 0.01},
        }

        score_result = compute_case_score_v2(results)

        # Degradation factor should be 0 when all match clean
        assert score_result.degradation_factor == pytest.approx(0.0, abs=0.001)
        # Absolute score should be 1% (the EER)
        assert score_result.case_score_absolute == pytest.approx(0.01, abs=0.001)

    def test_with_degradation(self):
        """Test V2 metrics with realistic degradation."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.03},  # 2% worse
            "clean_reverb": {"eer": 0.05},    # 4% worse
        }

        score_result = compute_case_score_v2(results)

        # Degradation should be positive
        assert score_result.degradation_factor > 0
        # Clean EER should be 1%
        assert score_result.clean_eer == pytest.approx(0.01, abs=1e-6)

    def test_v2_fixes_normalization_issue(self):
        """Test that V2 doesn't penalize models with good clean performance.

        The old formula (EER_degraded / EER_clean) penalized models with good
        clean performance:
          - Model A: Clean 5% -> Degraded 10% = Score 2.0
          - Model B: Clean 0.5% -> Degraded 5% = Score 10.0
        Model B was penalized despite being better (lower EERs).

        V2 uses absolute degradation, so better models get better scores.
        """
        # Model A: higher clean EER, moderate degradation
        results_a = {
            "clean_clean": {"eer": 0.05},
            "clean_codec_gsm": {"eer": 0.10},
        }

        # Model B: lower clean EER, same absolute degradation
        results_b = {
            "clean_clean": {"eer": 0.005},
            "clean_codec_gsm": {"eer": 0.055},
        }

        score_a = compute_case_score_v2(results_a)
        score_b = compute_case_score_v2(results_b)

        # Model B should have better (lower) absolute score
        assert score_b.case_score_absolute < score_a.case_score_absolute

        # Both should have similar degradation (5% increase)
        assert score_a.degradation_factor == pytest.approx(0.025, abs=0.01)  # (0.05 + 0.05) / 2
        assert score_b.degradation_factor == pytest.approx(0.025, abs=0.01)  # (0 + 0.05) / 2

    def test_category_degradation(self):
        """Test that per-category degradation is computed correctly."""
        results = {
            "clean_clean": {"eer": 0.01},
            "clean_codec_gsm": {"eer": 0.02},
            "clean_codec_ulaw": {"eer": 0.025},
        }

        score_result = compute_case_score_v2(results)

        # Clean category should have 0 degradation
        assert score_result.category_results["clean"]["degradation"] == 0.0

        # Codec category degradation = avg(0.02, 0.025) - 0.01 = 0.0125
        expected_deg = (0.02 + 0.025) / 2 - 0.01
        assert score_result.category_results["codec"]["degradation"] == pytest.approx(
            expected_deg, abs=1e-6
        )
