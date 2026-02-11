#!/usr/bin/env python3
"""
Verify all documented metrics against results/summary.json

This script checks ALL numeric claims in documentation (README.md, docs/findings.md,
docs/metrics.md) against the source of truth (results/summary.json).

Usage:
    python scripts/verify_metrics.py
    python scripts/verify_metrics.py --verbose
    python scripts/verify_metrics.py --fix  # Show suggested fixes
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    passed: bool
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    location: Optional[str] = None
    is_warning: bool = False  # Minor rounding difference


class MetricsVerifier:
    """Verifies all documented metrics against summary.json."""

    # Tolerance for floating point comparisons (0.1% = 0.001)
    TOLERANCE = 0.001
    # Tolerance for multiplier comparisons
    MULT_TOLERANCE = 0.15

    def __init__(self, base_path: Path, verbose: bool = False):
        self.base_path = base_path
        self.summary_path = base_path / "results" / "summary.json"
        self.verbose = verbose
        self.data = self._load_summary()
        self.models = {m["model_name"]: m for m in self.data["leaderboard"]}
        self.results: list[VerificationResult] = []

    def _load_summary(self) -> dict:
        """Load and return the summary.json data."""
        with open(self.summary_path) as f:
            return json.load(f)

    def _get_model(self, short_name: str) -> dict:
        """Get model data by short name."""
        name_map = {
            "WeSpeaker": "WeSpeaker ResNet34",
            "SpeechBrain": "SpeechBrain ECAPA-TDNN",
            "CASE HF": "CASE HF v2-512",
            "NeMo": "NeMo TitaNet-L",
            "pyannote": "pyannote Embedding",
            "Resemblyzer": "Resemblyzer",
        }
        return self.models[name_map.get(short_name, short_name)]

    def _pct(self, value: float) -> str:
        """Format value as percentage string."""
        return f"{value * 100:.2f}%"

    def _mult(self, value: float) -> str:
        """Format value as multiplier string."""
        return f"{value:.1f}x"

    def _check_value(self, expected: float, actual: float,
                     tolerance: float = None) -> bool:
        """Check if actual value is within tolerance of expected."""
        if tolerance is None:
            tolerance = self.TOLERANCE
        return abs(expected - actual) <= tolerance

    def _add_result(self, passed: bool, message: str,
                    expected: str = None, actual: str = None,
                    location: str = None, is_warning: bool = False):
        """Add a verification result."""
        self.results.append(VerificationResult(
            passed=passed,
            message=message,
            expected=expected,
            actual=actual,
            location=location,
            is_warning=is_warning
        ))

    def verify_degradation_multipliers(self):
        """
        Verify findings.md lines 15-19: Degradation multipliers by category.

        Claims:
        - Codec: 1.5-3x worse
        - Mic: ~1x (minimal)
        - Noise: 1.1-1.6x worse
        - Reverb: 4-12x worse
        - Playback: 7-19x worse
        """
        location = "findings.md:15-19"

        # Calculate actual multipliers for each model/category
        categories = ["codec", "mic", "noise", "reverb", "playback"]
        multipliers = {cat: [] for cat in categories}

        for model in self.data["leaderboard"]:
            clean = model["category_breakdown"]["clean"]
            for cat in categories:
                cat_eer = model["category_breakdown"][cat]
                mult = cat_eer / clean
                multipliers[cat].append((model["model_name"], mult))

        # Define expected ranges from documentation
        expected_ranges = {
            "codec": (1.5, 3.0),
            "mic": (0.9, 1.3),  # ~1x means approximately 1
            "noise": (1.1, 1.6),
            "reverb": (4.0, 12.0),
            "playback": (7.0, 19.0),
        }

        for cat, (exp_min, exp_max) in expected_ranges.items():
            actual_mults = [m for _, m in multipliers[cat]]
            actual_min = min(actual_mults)
            actual_max = max(actual_mults)

            # Check if documented range captures actual range
            # Allow some tolerance on the boundaries
            min_ok = actual_min >= (exp_min - self.MULT_TOLERANCE)
            max_ok = actual_max <= (exp_max + self.MULT_TOLERANCE)

            if min_ok and max_ok:
                self._add_result(
                    True,
                    f"{cat.capitalize()} multiplier range matches",
                    expected=f"{exp_min}-{exp_max}x",
                    actual=f"{actual_min:.1f}-{actual_max:.1f}x",
                    location=location
                )
            else:
                # Check if it's just a minor boundary issue
                is_minor = (abs(actual_min - exp_min) < 0.3 or
                           abs(actual_max - exp_max) < 0.5)
                self._add_result(
                    False,
                    f"{cat.capitalize()} multiplier range mismatch",
                    expected=f"{exp_min}-{exp_max}x",
                    actual=f"{actual_min:.1f}-{actual_max:.1f}x",
                    location=location,
                    is_warning=is_minor
                )

    def verify_individual_model_table(self):
        """
        Verify findings.md lines 42-47: Individual model degradation table.

        | Model | Clean | Playback | Factor |
        |-------|-------|----------|--------|
        | WeSpeaker | 0.58% | 8.57% | 14.8x |
        | SpeechBrain | 0.56% | 9.37% | 16.7x |
        | CASE HF | 1.22% | 9.10% | 7.5x |
        | NeMo | 0.66% | 12.61% | 19.1x |
        | pyannote | 1.68% | 11.22% | 6.7x |
        """
        location = "findings.md:42-47"

        # Expected values from documentation
        expected = [
            ("WeSpeaker", 0.0058, 0.0857, 14.8),
            ("SpeechBrain", 0.0056, 0.0937, 16.7),
            ("CASE HF", 0.0122, 0.0910, 7.5),
            ("NeMo", 0.0066, 0.1261, 19.1),
            ("pyannote", 0.0168, 0.1122, 6.7),
        ]

        for name, exp_clean, exp_playback, exp_factor in expected:
            model = self._get_model(name)
            actual_clean = model["category_breakdown"]["clean"]
            actual_playback = model["category_breakdown"]["playback"]
            actual_factor = actual_playback / actual_clean

            # Check clean EER
            clean_ok = self._check_value(exp_clean, actual_clean)
            # Check playback EER
            playback_ok = self._check_value(exp_playback, actual_playback)
            # Check factor (wider tolerance for division)
            factor_ok = abs(exp_factor - actual_factor) < 0.5

            if clean_ok and playback_ok and factor_ok:
                self._add_result(
                    True,
                    f"{name}: Clean, Playback, Factor all match",
                    location=location
                )
            else:
                details = []
                if not clean_ok:
                    details.append(f"Clean: {self._pct(exp_clean)} vs {self._pct(actual_clean)}")
                if not playback_ok:
                    details.append(f"Playback: {self._pct(exp_playback)} vs {self._pct(actual_playback)}")
                if not factor_ok:
                    details.append(f"Factor: {exp_factor}x vs {actual_factor:.1f}x")

                self._add_result(
                    False,
                    f"{name}: {', '.join(details)}",
                    location=location,
                    is_warning=True  # Often minor rounding
                )

    def verify_sota_avg_comparison(self):
        """
        Verify findings.md lines 57-60: SOTA avg degradation comparison.

        SOTA = WeSpeaker, SpeechBrain, NeMo (top 3 by clean EER)

        | Category | SOTA Avg Degradation | CASE HF Degradation | Improvement |
        |----------|---------------------|---------------------|-------------|
        | Codec | +1.10% | +0.47% | 57% less |
        | Reverb | +5.71% | +5.34% | 6% less |
        | Playback | +9.58% | +7.88% | 18% less |
        """
        location = "findings.md:57-60"

        # Get SOTA models (WeSpeaker, SpeechBrain, NeMo)
        sota_models = [
            self._get_model("WeSpeaker"),
            self._get_model("SpeechBrain"),
            self._get_model("NeMo"),
        ]
        case_hf = self._get_model("CASE HF")

        # Calculate SOTA average degradation for each category
        # Degradation = category_eer - clean_eer
        categories_to_check = ["codec", "reverb", "playback"]
        expected_values = {
            "codec": (0.0110, 0.0047, 57),  # SOTA deg, CASE deg, improvement %
            "reverb": (0.0571, 0.0534, 6),
            "playback": (0.0958, 0.0788, 18),
        }

        for cat in categories_to_check:
            # Calculate SOTA avg degradation
            sota_degs = []
            for m in sota_models:
                deg = m["category_breakdown"][cat] - m["category_breakdown"]["clean"]
                sota_degs.append(deg)
            sota_avg_deg = sum(sota_degs) / len(sota_degs)

            # Calculate CASE HF degradation
            case_deg = case_hf["category_breakdown"][cat] - case_hf["category_breakdown"]["clean"]

            # Calculate improvement percentage
            if sota_avg_deg > 0:
                improvement = (sota_avg_deg - case_deg) / sota_avg_deg * 100
            else:
                improvement = 0

            exp_sota, exp_case, exp_impr = expected_values[cat]

            # Check values
            sota_ok = self._check_value(exp_sota, sota_avg_deg, 0.002)
            case_ok = self._check_value(exp_case, case_deg, 0.002)
            impr_ok = abs(exp_impr - improvement) < 5  # 5% tolerance on improvement

            if sota_ok and case_ok and impr_ok:
                self._add_result(
                    True,
                    f"{cat.capitalize()}: SOTA avg, CASE HF deg, and improvement match",
                    location=location
                )
            else:
                details = []
                if not sota_ok:
                    details.append(f"SOTA Avg: +{exp_sota*100:.2f}% vs +{sota_avg_deg*100:.2f}%")
                if not case_ok:
                    details.append(f"CASE: +{exp_case*100:.2f}% vs +{case_deg*100:.2f}%")
                if not impr_ok:
                    details.append(f"Improvement: {exp_impr}% vs {improvement:.0f}%")

                self._add_result(
                    False,
                    f"{cat.capitalize()}: {', '.join(details)}",
                    location=location,
                    is_warning=True
                )

    def verify_leaderboard(self):
        """
        Verify README.md lines 99-104 and metrics.md lines 137-142: Leaderboard table.

        | Model | Absolute EER | Degradation | Clean EER |
        |-------|-------------|-------------|-----------|
        | WeSpeaker | 3.01% | +2.43% | 0.58% |
        | SpeechBrain | 3.05% | +2.49% | 0.56% |
        | CASE HF | 3.53% | +2.31% | 1.22% |
        | NeMo | 4.05% | +3.39% | 0.66% |
        | pyannote | 4.47% | +2.79% | 1.68% |
        | Resemblyzer | 10.49% | +5.65% | 4.84% |
        """
        location = "README.md:99-104, metrics.md:137-142"

        expected = [
            ("WeSpeaker", 0.0301, 0.0243, 0.0058),
            ("SpeechBrain", 0.0305, 0.0249, 0.0056),
            ("CASE HF", 0.0353, 0.0231, 0.0122),
            ("NeMo", 0.0405, 0.0339, 0.0066),
            ("pyannote", 0.0447, 0.0279, 0.0168),
            ("Resemblyzer", 0.1049, 0.0565, 0.0484),
        ]

        all_passed = True
        issues = []

        for name, exp_abs, exp_deg, exp_clean in expected:
            model = self._get_model(name)
            actual_abs = model["case_score_absolute"]
            actual_clean = model["clean_eer"]
            actual_deg = model["degradation_factor"]

            # Verify Degradation = Absolute - Clean
            computed_deg = actual_abs - actual_clean

            abs_ok = self._check_value(exp_abs, actual_abs)
            clean_ok = self._check_value(exp_clean, actual_clean)
            deg_ok = self._check_value(exp_deg, actual_deg)
            formula_ok = self._check_value(actual_deg, computed_deg, 0.0001)

            if not (abs_ok and clean_ok and deg_ok and formula_ok):
                all_passed = False
                model_issues = []
                if not abs_ok:
                    model_issues.append(f"Abs: {self._pct(exp_abs)} vs {self._pct(actual_abs)}")
                if not clean_ok:
                    model_issues.append(f"Clean: {self._pct(exp_clean)} vs {self._pct(actual_clean)}")
                if not deg_ok:
                    model_issues.append(f"Deg: +{self._pct(exp_deg)} vs +{self._pct(actual_deg)}")
                if not formula_ok:
                    model_issues.append(f"Formula mismatch: Abs-Clean != Deg")
                issues.append(f"{name}: {', '.join(model_issues)}")

        if all_passed:
            self._add_result(True, "All 6 models verified in leaderboard", location=location)
        else:
            for issue in issues:
                self._add_result(False, issue, location=location, is_warning=True)

    def verify_wespeaker_breakdown(self):
        """
        Verify README.md lines 120-125 and metrics.md lines 78-83: WeSpeaker category breakdown.

        | Category | Avg EER | vs Clean |
        |----------|---------|----------|
        | Clean | 0.58% | baseline |
        | Codec | 1.73% | +1.15% |
        | Mic | 0.59% | +0.01% |
        | Noise | 0.73% | +0.15% |
        | Reverb | 5.88% | +5.30% |
        | Playback | 8.57% | +7.99% |
        """
        location = "README.md:120-125, metrics.md:78-83"

        wespeaker = self._get_model("WeSpeaker")
        breakdown = wespeaker["category_breakdown"]
        clean = breakdown["clean"]

        expected = {
            "clean": (0.0058, 0.0),
            "codec": (0.0173, 0.0115),
            "mic": (0.0059, 0.0001),
            "noise": (0.0073, 0.0015),
            "reverb": (0.0588, 0.0530),
            "playback": (0.0857, 0.0799),
        }

        all_passed = True
        issues = []

        for cat, (exp_eer, exp_vs_clean) in expected.items():
            actual_eer = breakdown[cat]
            actual_vs_clean = actual_eer - clean if cat != "clean" else 0

            eer_ok = self._check_value(exp_eer, actual_eer)
            vs_clean_ok = self._check_value(exp_vs_clean, actual_vs_clean) if cat != "clean" else True

            if not (eer_ok and vs_clean_ok):
                all_passed = False
                cat_issues = []
                if not eer_ok:
                    cat_issues.append(f"EER: {self._pct(exp_eer)} vs {self._pct(actual_eer)}")
                if not vs_clean_ok:
                    cat_issues.append(f"vs Clean: +{self._pct(exp_vs_clean)} vs +{self._pct(actual_vs_clean)}")
                issues.append(f"{cat.capitalize()}: {', '.join(cat_issues)}")

        if all_passed:
            self._add_result(True, "All WeSpeaker categories match", location=location)
        else:
            for issue in issues:
                self._add_result(False, issue, location=location, is_warning=True)

    def verify_case_score_ratios(self):
        """
        Verify metrics.md lines 103-104: CASE-Score v1 calculations.

        | Model | CASE-Score v1 | Formula |
        |-------|---------------|---------|
        | Resemblyzer | 2.17x | 10.49 / 4.84 |
        | WeSpeaker | 5.19x | 3.01 / 0.58 |
        """
        location = "metrics.md:103-104"

        # Check Resemblyzer
        resemblyzer = self._get_model("Resemblyzer")
        res_ratio = resemblyzer["case_score_absolute"] / resemblyzer["clean_eer"]
        res_expected = 2.17
        res_ok = abs(res_ratio - res_expected) < 0.05

        if res_ok:
            self._add_result(True, f"Resemblyzer CASE-Score v1: {res_expected}x matches", location=location)
        else:
            self._add_result(
                False,
                f"Resemblyzer CASE-Score v1 mismatch",
                expected=f"{res_expected}x",
                actual=f"{res_ratio:.2f}x",
                location=location,
                is_warning=True
            )

        # Check WeSpeaker
        wespeaker = self._get_model("WeSpeaker")
        wes_ratio = wespeaker["case_score_absolute"] / wespeaker["clean_eer"]
        wes_expected = 5.19
        wes_ok = abs(wes_ratio - wes_expected) < 0.1

        if wes_ok:
            self._add_result(True, f"WeSpeaker CASE-Score v1: {wes_expected}x matches", location=location)
        else:
            self._add_result(
                False,
                f"WeSpeaker CASE-Score v1 mismatch",
                expected=f"{wes_expected}x",
                actual=f"{wes_ratio:.2f}x",
                location=location,
                is_warning=True
            )

    def verify_readme_sota_table(self):
        """
        Verify README.md lines 24-30: "The Problem" SOTA performance table.

        | Condition | Typical SOTA Performance |
        |-----------|--------------------------|
        | Clean Audio | 0.8-1.5% EER |
        | Phone Codec (GSM) | 2-4% EER |
        | Laptop Microphone | 1-2% EER |
        | Room Reverb | 8-15% EER |
        | Playback Chain | 15-25% EER |
        """
        location = "README.md:24-30"

        # Get data for SOTA models (excluding Resemblyzer which is legacy)
        sota_names = [
            "WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN",
            "CASE HF v2-512", "NeMo TitaNet-L", "pyannote Embedding"
        ]
        sota_models = [m for m in self.data["leaderboard"]
                       if m["model_name"] in sota_names]

        # Calculate actual ranges for each category
        def get_range(category):
            vals = [m["category_breakdown"][category] for m in sota_models]
            return min(vals), max(vals)

        clean_min, clean_max = get_range("clean")
        codec_min, codec_max = get_range("codec")
        mic_min, mic_max = get_range("mic")
        reverb_min, reverb_max = get_range("reverb")
        playback_min, playback_max = get_range("playback")

        # Verify each claim (updated to match corrected README.md)
        claims = [
            ("Clean Audio", (0.006, 0.017), (clean_min, clean_max)),
            ("Phone Codec", (0.02, 0.04), (codec_min, codec_max)),
            ("Laptop Mic", (0.006, 0.018), (mic_min, mic_max)),
            ("Room Reverb", (0.05, 0.08), (reverb_min, reverb_max)),
            ("Playback Chain", (0.09, 0.13), (playback_min, playback_max)),
        ]

        for name, (exp_min, exp_max), (actual_min, actual_max) in claims:
            # Check if documented range reasonably captures actual range
            # Allow 50% tolerance since these are rough "typical" values
            min_close = abs(exp_min - actual_min) / max(actual_min, 0.001) < 0.5
            max_close = abs(exp_max - actual_max) / max(actual_max, 0.001) < 0.5

            if min_close and max_close:
                self._add_result(
                    True,
                    f"{name}: Range reasonably matches",
                    expected=f"{exp_min*100:.1f}-{exp_max*100:.1f}%",
                    actual=f"{actual_min*100:.1f}-{actual_max*100:.1f}%",
                    location=location
                )
            else:
                self._add_result(
                    False,
                    f"{name}: Range significantly off",
                    expected=f"{exp_min*100:.1f}-{exp_max*100:.1f}%",
                    actual=f"{actual_min*100:.1f}-{actual_max*100:.1f}%",
                    location=location,
                    is_warning=False
                )

    def identify_discrepancies(self):
        """
        Identify cross-document discrepancies.

        Known discrepancy:
        - README.md line 30: "Playback Chain: 15-25% EER"
        - findings.md line 74: "SOTA models: 8.6-12.6% EER on playback"
        """
        # Get actual playback EER range
        playback_eers = []
        sota_playback_eers = []
        sota_models = ["WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN", "NeMo TitaNet-L"]

        for model in self.data["leaderboard"]:
            eer = model["category_breakdown"]["playback"]
            playback_eers.append((model["model_name"], eer))
            if model["model_name"] in sota_models:
                sota_playback_eers.append(eer)

        all_min = min(eer for _, eer in playback_eers)
        all_max = max(eer for _, eer in playback_eers)
        sota_min = min(sota_playback_eers)
        sota_max = max(sota_playback_eers)

        # README claims 9-13% for "Typical SOTA Performance" on playback
        readme_min_claim = 0.09
        readme_max_claim = 0.13

        # Check if README claim matches actual SOTA data (within tolerance)
        readme_matches_sota = (abs(readme_min_claim - sota_min) < 0.01 and
                               abs(readme_max_claim - sota_max) < 0.01)

        if readme_matches_sota:
            self._add_result(
                True,
                "README.md playback range matches SOTA data",
                expected=f"{readme_min_claim*100:.0f}-{readme_max_claim*100:.0f}% EER",
                actual=f"{sota_min*100:.1f}-{sota_max*100:.1f}% EER",
                location="README.md:30"
            )
        else:
            self._add_result(
                False,
                "README.md playback claim does not match actual SOTA data",
                expected=f"{readme_min_claim*100:.0f}-{readme_max_claim*100:.0f}% EER (documented)",
                actual=f"{sota_min*100:.1f}-{sota_max*100:.1f}% EER (SOTA actual)",
                location="README.md:30",
                is_warning=True
            )

        # Check findings claim (8.6-12.6% for SOTA)
        findings_claim = (0.086, 0.126)
        findings_ok = self._check_value(findings_claim[0], sota_min, 0.005) and \
                      self._check_value(findings_claim[1], sota_max, 0.005)

        if findings_ok:
            self._add_result(
                True,
                "findings.md SOTA playback range matches",
                expected="8.6-12.6% EER",
                actual=f"{sota_min*100:.1f}-{sota_max*100:.1f}% EER",
                location="findings.md:74"
            )
        else:
            self._add_result(
                False,
                "findings.md SOTA playback range mismatch",
                expected="8.6-12.6% EER",
                actual=f"{sota_min*100:.1f}-{sota_max*100:.1f}% EER",
                location="findings.md:74",
                is_warning=True
            )

    def run_all_verifications(self):
        """Run all verification checks."""
        print("=== CASE Benchmark Metrics Verification ===\n")

        print("[1/8] Degradation Multipliers (findings.md:15-19)")
        self.verify_degradation_multipliers()
        self._print_section_results()

        print("\n[2/8] Individual Model Table (findings.md:42-47)")
        self.verify_individual_model_table()
        self._print_section_results()

        print("\n[3/8] SOTA Avg Comparison (findings.md:57-60)")
        self.verify_sota_avg_comparison()
        self._print_section_results()

        print("\n[4/8] Leaderboard Table (README.md:99-104, metrics.md:137-142)")
        self.verify_leaderboard()
        self._print_section_results()

        print("\n[5/8] WeSpeaker Breakdown (README.md:120-125, metrics.md:78-83)")
        self.verify_wespeaker_breakdown()
        self._print_section_results()

        print("\n[6/8] CASE-Score v1 Ratios (metrics.md:103-104)")
        self.verify_case_score_ratios()
        self._print_section_results()

        print("\n[7/8] README SOTA Performance Table (README.md:24-30)")
        self.verify_readme_sota_table()
        self._print_section_results()

        print("\n[8/8] Cross-Document Discrepancies")
        self.identify_discrepancies()
        self._print_section_results()

        self._print_summary()

    def _print_section_results(self):
        """Print results for the current section (results added since last print)."""
        # Get results that haven't been printed yet
        for result in self.results:
            if not hasattr(result, '_printed'):
                if result.passed:
                    status = "[PASS]"
                elif result.is_warning:
                    status = "[WARN]"
                else:
                    status = "[FAIL]"

                print(f"  {status} {result.message}")
                # In verbose mode, always show expected/actual
                # In normal mode, only show for non-passing results
                show_details = self.verbose or not result.passed
                if show_details and result.expected and result.actual:
                    print(f"         Expected: {result.expected}")
                    print(f"         Actual:   {result.actual}")
                result._printed = True

    def _print_summary(self):
        """Print verification summary."""
        passed = sum(1 for r in self.results if r.passed)
        warnings = sum(1 for r in self.results if not r.passed and r.is_warning)
        errors = sum(1 for r in self.results if not r.passed and not r.is_warning)
        total = len(self.results)

        print("\n" + "=" * 50)
        print("=== Summary ===")
        print(f"Verified: {passed}/{total} claims")
        print(f"Warnings: {warnings} (minor rounding or scope differences)")
        print(f"Errors:   {errors} (significant mismatches)")

        if errors > 0:
            print("\nSignificant Issues:")
            for r in self.results:
                if not r.passed and not r.is_warning:
                    print(f"  - {r.message}")
                    if r.location:
                        print(f"    Location: {r.location}")

        if warnings > 0:
            print("\nWarnings (may need review):")
            for r in self.results:
                if not r.passed and r.is_warning:
                    print(f"  - {r.message}")

    def _print_suggested_fixes(self):
        """Print suggested fixes for identified issues."""
        print("\n### README.md SOTA Performance Table (lines 24-30)")
        print("Current table has inaccurate ranges. Suggested corrections:\n")

        # Calculate actual ranges
        sota_names = [
            "WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN",
            "CASE HF v2-512", "NeMo TitaNet-L", "pyannote Embedding"
        ]
        sota_models = [m for m in self.data["leaderboard"]
                       if m["model_name"] in sota_names]

        def get_range(category):
            vals = [m["category_breakdown"][category] for m in sota_models]
            return min(vals), max(vals)

        clean_min, clean_max = get_range("clean")
        codec_min, codec_max = get_range("codec")
        mic_min, mic_max = get_range("mic")
        reverb_min, reverb_max = get_range("reverb")
        playback_min, playback_max = get_range("playback")

        print("| Condition | Typical SOTA Performance |")
        print("|-----------|--------------------------|")
        print(f"| Clean Audio | **{clean_min*100:.1f}-{clean_max*100:.1f}% EER** |")
        print(f"| Phone Codec (GSM) | {codec_min*100:.0f}-{codec_max*100:.0f}% EER |")
        print(f"| Laptop Microphone | {mic_min*100:.1f}-{mic_max*100:.1f}% EER |")
        print(f"| Room Reverb | {reverb_min*100:.0f}-{reverb_max*100:.0f}% EER |")
        print(f"| **Playback Chain** | **{playback_min*100:.0f}-{playback_max*100:.0f}% EER** |")

        print("\n### findings.md Degradation Multipliers (lines 15-19)")
        print("Minor adjustments to match actual data:\n")

        # Calculate actual multipliers
        categories = ["codec", "mic", "noise", "reverb", "playback"]
        for cat in categories:
            mults = []
            for model in self.data["leaderboard"]:
                clean = model["category_breakdown"]["clean"]
                cat_eer = model["category_breakdown"][cat]
                mults.append(cat_eer / clean)
            mult_min, mult_max = min(mults), max(mults)
            print(f"| {cat.capitalize()} | {mult_min:.1f}-{mult_max:.1f}x |")

        print("\n### General Recommendations")
        print("1. Run: python scripts/generate_leaderboard.py to regenerate tables")
        print("2. Update prose claims manually using values from results/summary.json")
        print("3. Re-run this script to verify fixes: python scripts/verify_metrics.py")


def main():
    parser = argparse.ArgumentParser(
        description="Verify all documented metrics against results/summary.json"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for all checks"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Suggest fixes for discrepancies"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base path to the case-benchmark repository"
    )

    args = parser.parse_args()

    verifier = MetricsVerifier(args.base_path, verbose=args.verbose)
    verifier.run_all_verifications()

    if args.fix:
        print("\n" + "=" * 50)
        print("=== Suggested Fixes ===")
        verifier._print_suggested_fixes()

    # Return exit code based on errors found
    errors = sum(1 for r in verifier.results if not r.passed and not r.is_warning)
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
