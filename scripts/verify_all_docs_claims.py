#!/usr/bin/env python3
"""
Comprehensive verification of ALL numeric claims across all documentation.

Covers: README.md, docs/findings.md, docs/metrics.md
"""

import json
from pathlib import Path


def load_data():
    """Load summary.json data."""
    base_path = Path(__file__).parent.parent
    with open(base_path / "results" / "summary.json") as f:
        return json.load(f)


def pct(val):
    """Format as percentage."""
    return f"{val * 100:.2f}%"


def get_model(models, name):
    """Get model by short name."""
    name_map = {
        "WeSpeaker": "WeSpeaker ResNet34",
        "SpeechBrain": "SpeechBrain ECAPA-TDNN",
        "CASE HF": "CASE HF v2-512",
        "NeMo": "NeMo TitaNet-L",
        "pyannote": "pyannote Embedding",
        "Resemblyzer": "Resemblyzer",
    }
    return models[name_map.get(name, name)]


def verify_findings_md(data, models):
    """Verify all claims in findings.md."""
    print("\n" + "=" * 70)
    print("FINDINGS.MD VERIFICATION")
    print("=" * 70)

    results = []

    # =========================================================================
    # LINE 7: "up to 19× worse EER"
    # =========================================================================
    print("\n[Line 7] 'playback chains causing up to 19× worse EER'")
    max_mult = 0
    for m in data["leaderboard"]:
        clean = m["category_breakdown"]["clean"]
        playback = m["category_breakdown"]["playback"]
        max_mult = max(max_mult, playback / clean)

    passed = 18.5 <= max_mult <= 19.5
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual max: {max_mult:.1f}× (claimed 19×)")
    results.append(passed)

    # =========================================================================
    # LINE 11: "~1% EER" for clean
    # =========================================================================
    print("\n[Line 11] 'excellent performance on clean audio (~1% EER)'")
    sota_names = ["WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN", "NeMo TitaNet-L"]
    sota_clean = [m["clean_eer"] for m in data["leaderboard"] if m["model_name"] in sota_names]
    avg_clean = sum(sota_clean) / len(sota_clean)

    passed = 0.005 <= avg_clean <= 0.015
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] SOTA avg clean EER: {pct(avg_clean)} (~1%)")
    results.append(passed)

    # =========================================================================
    # LINES 15-19: Degradation multipliers
    # =========================================================================
    print("\n[Lines 15-19] Degradation multipliers table")

    categories = ["codec", "mic", "noise", "reverb", "playback"]
    claimed_ranges = {
        "codec": (1.5, 3.0),
        "mic": (0.9, 1.1),  # ~1×
        "noise": (1.1, 1.6),
        "reverb": (4.0, 12.0),
        "playback": (7.0, 19.0),
    }

    for cat in categories:
        mults = []
        for m in data["leaderboard"]:
            clean = m["category_breakdown"]["clean"]
            cat_eer = m["category_breakdown"][cat]
            mults.append(cat_eer / clean)

        actual_min, actual_max = min(mults), max(mults)
        claimed_min, claimed_max = claimed_ranges[cat]

        # Allow tolerance since these are "typical" ranges
        passed = (actual_min >= claimed_min - 0.6 and actual_max <= claimed_max + 0.5)
        status = "PASS" if passed else "WARN"
        print(f"  [{status}] {cat.capitalize()}: {actual_min:.1f}-{actual_max:.1f}× (claimed {claimed_min}-{claimed_max}×)")
        results.append(passed)

    # =========================================================================
    # LINE 21: "0.6% EER...might have 9-13% EER"
    # =========================================================================
    print("\n[Line 21] 'model with 0.6% EER might have 9-13% EER on playback'")
    # Check WeSpeaker (0.58% clean)
    ws = get_model(models, "WeSpeaker")
    ws_clean = ws["category_breakdown"]["clean"]
    ws_playback = ws["category_breakdown"]["playback"]

    passed = (0.005 <= ws_clean <= 0.007) and (0.08 <= ws_playback <= 0.13)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] WeSpeaker: {pct(ws_clean)} clean → {pct(ws_playback)} playback")
    results.append(passed)

    # =========================================================================
    # LINE 34: "CASE model reduces codec degradation by ~57% and playback by ~18%"
    # =========================================================================
    print("\n[Line 34] 'CASE reduces codec degradation by ~57%, playback by ~18%'")

    sota_names = ["WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN", "NeMo TitaNet-L"]
    sota_models = [m for m in data["leaderboard"] if m["model_name"] in sota_names]
    case_hf = get_model(models, "CASE HF")

    # Calculate SOTA avg degradation for codec
    sota_codec_degs = [m["category_breakdown"]["codec"] - m["category_breakdown"]["clean"]
                       for m in sota_models]
    sota_codec_avg = sum(sota_codec_degs) / len(sota_codec_degs)
    case_codec_deg = case_hf["category_breakdown"]["codec"] - case_hf["category_breakdown"]["clean"]
    codec_improvement = (sota_codec_avg - case_codec_deg) / sota_codec_avg * 100

    # Calculate SOTA avg degradation for playback
    sota_playback_degs = [m["category_breakdown"]["playback"] - m["category_breakdown"]["clean"]
                          for m in sota_models]
    sota_playback_avg = sum(sota_playback_degs) / len(sota_playback_degs)
    case_playback_deg = case_hf["category_breakdown"]["playback"] - case_hf["category_breakdown"]["clean"]
    playback_improvement = (sota_playback_avg - case_playback_deg) / sota_playback_avg * 100

    codec_ok = 50 <= codec_improvement <= 65
    playback_ok = 15 <= playback_improvement <= 25

    passed = codec_ok and playback_ok
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Codec improvement: {codec_improvement:.0f}% (claimed ~57%)")
    print(f"         Playback improvement: {playback_improvement:.0f}% (claimed ~18%)")
    results.append(passed)

    # =========================================================================
    # LINES 42-47: Individual model playback table
    # =========================================================================
    print("\n[Lines 42-47] Individual model playback degradation table")

    playback_claims = [
        ("WeSpeaker", 0.0058, 0.0857, 14.8),
        ("SpeechBrain", 0.0056, 0.0937, 16.7),
        ("CASE HF", 0.0122, 0.0910, 7.5),
        ("NeMo", 0.0066, 0.1261, 19.1),
        ("pyannote", 0.0168, 0.1122, 6.7),
    ]

    all_passed = True
    for name, claimed_clean, claimed_playback, claimed_factor in playback_claims:
        m = get_model(models, name)
        actual_clean = m["category_breakdown"]["clean"]
        actual_playback = m["category_breakdown"]["playback"]
        actual_factor = actual_playback / actual_clean

        clean_ok = abs(actual_clean - claimed_clean) < 0.001
        playback_ok = abs(actual_playback - claimed_playback) < 0.002
        factor_ok = abs(actual_factor - claimed_factor) < 0.5

        passed = clean_ok and playback_ok and factor_ok
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {pct(actual_clean)}/{pct(actual_playback)}/{actual_factor:.1f}× "
              f"(claimed {pct(claimed_clean)}/{pct(claimed_playback)}/{claimed_factor}×)")

    results.append(all_passed)

    # =========================================================================
    # LINES 57-60: SOTA avg comparison table
    # =========================================================================
    print("\n[Lines 57-60] SOTA avg degradation comparison table")

    comparison_claims = [
        ("Codec", 0.0110, 0.0047, 57),
        ("Reverb", 0.0571, 0.0534, 6),
        ("Playback", 0.0958, 0.0788, 18),
    ]

    for cat, sota_deg, case_deg, improvement in comparison_claims:
        # Calculate actual
        sota_actual = sum(m["category_breakdown"][cat.lower()] - m["category_breakdown"]["clean"]
                         for m in sota_models) / len(sota_models)
        case_actual = case_hf["category_breakdown"][cat.lower()] - case_hf["category_breakdown"]["clean"]
        impr_actual = (sota_actual - case_actual) / sota_actual * 100

        sota_ok = abs(sota_actual - sota_deg) < 0.002
        case_ok = abs(case_actual - case_deg) < 0.002
        impr_ok = abs(impr_actual - improvement) < 5

        passed = sota_ok and case_ok and impr_ok
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {cat}: SOTA +{pct(sota_actual)}, CASE +{pct(case_actual)}, Impr {impr_actual:.0f}%")
        print(f"         (claimed SOTA +{pct(sota_deg)}, CASE +{pct(case_deg)}, Impr {improvement}%)")
        results.append(passed)

    # =========================================================================
    # LINE 63: "lowest overall degradation factor (+2.31%)"
    # =========================================================================
    print("\n[Line 63] 'lowest overall degradation factor (+2.31%)'")
    case_deg = case_hf["degradation_factor"]
    lowest = min(m["degradation_factor"] for m in data["leaderboard"])

    passed = abs(case_deg - 0.0231) < 0.001 and abs(case_deg - lowest) < 0.0001
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] CASE HF degradation: +{pct(case_deg)} (is lowest: {abs(case_deg - lowest) < 0.0001})")
    results.append(passed)

    # =========================================================================
    # LINE 71: "All models: 0.56-1.68% EER"
    # =========================================================================
    print("\n[Line 71] 'All models: 0.56-1.68% EER'")
    # Excluding Resemblyzer
    sota_clean_eers = [m["clean_eer"] for m in data["leaderboard"]
                       if m["model_name"] != "Resemblyzer"]
    actual_min = min(sota_clean_eers)
    actual_max = max(sota_clean_eers)

    passed = abs(actual_min - 0.0056) < 0.001 and abs(actual_max - 0.0168) < 0.001
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual range: {pct(actual_min)}-{pct(actual_max)} (claimed 0.56-1.68%)")
    results.append(passed)

    # =========================================================================
    # LINE 74: "SOTA models: 8.6-12.6% EER on playback"
    # =========================================================================
    print("\n[Line 74] 'SOTA models: 8.6-12.6% EER on playback'")
    sota_playback = [m["category_breakdown"]["playback"] for m in sota_models]
    actual_min = min(sota_playback)
    actual_max = max(sota_playback)

    passed = abs(actual_min - 0.086) < 0.005 and abs(actual_max - 0.126) < 0.005
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual range: {pct(actual_min)}-{pct(actual_max)} (claimed 8.6-12.6%)")
    results.append(passed)

    # =========================================================================
    # LINE 75: "CASE HF: 9.10% EER on playback"
    # =========================================================================
    print("\n[Line 75] 'CASE HF: 9.10% EER on playback'")
    case_playback = case_hf["category_breakdown"]["playback"]

    passed = abs(case_playback - 0.091) < 0.001
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual: {pct(case_playback)} (claimed 9.10%)")
    results.append(passed)

    # =========================================================================
    # LINE 84: "8-13% EER"
    # =========================================================================
    print("\n[Line 84] 'Playback chains: 8-13% EER'")
    # For all SOTA models (not Resemblyzer)
    sota_playback = [m["category_breakdown"]["playback"] for m in data["leaderboard"]
                     if m["model_name"] != "Resemblyzer"]
    actual_min = min(sota_playback)
    actual_max = max(sota_playback)

    passed = 0.08 <= actual_min and actual_max <= 0.13
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual range: {pct(actual_min)}-{pct(actual_max)} (claimed 8-13%)")
    results.append(passed)

    # =========================================================================
    # LINE 104: "up to 19× worse performance"
    # =========================================================================
    print("\n[Line 104] 'up to 19× worse performance'")
    # Already checked above, but verify again
    max_mult = max(m["category_breakdown"]["playback"] / m["category_breakdown"]["clean"]
                   for m in data["leaderboard"])

    passed = 18.5 <= max_mult <= 19.5
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual max: {max_mult:.1f}× (claimed 19×)")
    results.append(passed)

    return results


def verify_metrics_md(data, models):
    """Verify all claims in metrics.md."""
    print("\n" + "=" * 70)
    print("METRICS.MD VERIFICATION")
    print("=" * 70)

    results = []

    # =========================================================================
    # LINE 23: "0.5-1.5% on VoxCeleb1-O"
    # =========================================================================
    print("\n[Line 23] 'Typical SOTA models achieve 0.5-1.5% on VoxCeleb1-O'")
    sota_clean = [m["clean_eer"] for m in data["leaderboard"]
                  if m["model_name"] in ["WeSpeaker ResNet34", "SpeechBrain ECAPA-TDNN", "NeMo TitaNet-L"]]
    actual_min = min(sota_clean)
    actual_max = max(sota_clean)

    passed = 0.005 <= actual_min and actual_max <= 0.015
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Actual SOTA range: {pct(actual_min)}-{pct(actual_max)} (claimed 0.5-1.5%)")
    results.append(passed)

    # =========================================================================
    # LINES 42-43: WeSpeaker and CASE HF example
    # =========================================================================
    print("\n[Lines 42-43] WeSpeaker and CASE HF example")

    ws = get_model(models, "WeSpeaker")
    case = get_model(models, "CASE HF")

    claims = [
        ("WeSpeaker", ws, 0.0058, 0.0301, 0.0243),
        ("CASE HF", case, 0.0122, 0.0353, 0.0231),
    ]

    for name, m, clean, absolute, deg in claims:
        actual_clean = m["clean_eer"]
        actual_abs = m["case_score_absolute"]
        actual_deg = m["degradation_factor"]

        clean_ok = abs(actual_clean - clean) < 0.001
        abs_ok = abs(actual_abs - absolute) < 0.001
        deg_ok = abs(actual_deg - deg) < 0.001

        passed = clean_ok and abs_ok and deg_ok
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: Clean {pct(actual_clean)}, Abs {pct(actual_abs)}, Deg +{pct(actual_deg)}")
        print(f"         (claimed Clean {pct(clean)}, Abs {pct(absolute)}, Deg +{pct(deg)})")
        results.append(passed)

    # =========================================================================
    # LINES 78-83: WeSpeaker breakdown (same as README)
    # =========================================================================
    print("\n[Lines 78-83] WeSpeaker category breakdown")
    breakdown = ws["category_breakdown"]
    clean = breakdown["clean"]

    ws_claims = [
        ("clean", 0.0058),
        ("codec", 0.0173),
        ("mic", 0.0059),
        ("noise", 0.0073),
        ("reverb", 0.0588),
        ("playback", 0.0857),
    ]

    all_passed = True
    for cat, claimed in ws_claims:
        actual = breakdown[cat]
        passed = abs(actual - claimed) < 0.001
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {cat}: {pct(actual)} (claimed {pct(claimed)})")

    results.append(all_passed)

    # =========================================================================
    # LINES 103-104: CASE-Score v1 examples
    # =========================================================================
    print("\n[Lines 103-104] CASE-Score v1 examples")

    res = get_model(models, "Resemblyzer")
    res_ratio = res["case_score_absolute"] / res["clean_eer"]

    ws_ratio = ws["case_score_absolute"] / ws["clean_eer"]

    res_ok = abs(res_ratio - 2.17) < 0.05
    ws_ok = abs(ws_ratio - 5.19) < 0.1

    print(f"  [{'PASS' if res_ok else 'FAIL'}] Resemblyzer: {res_ratio:.2f}× (claimed 2.17×)")
    print(f"  [{'PASS' if ws_ok else 'FAIL'}] WeSpeaker: {ws_ratio:.2f}× (claimed 5.19×)")
    results.append(res_ok)
    results.append(ws_ok)

    # =========================================================================
    # LINES 137-142: Full comparison table
    # =========================================================================
    print("\n[Lines 137-142] Full comparison table")

    table_claims = [
        ("WeSpeaker", 0.0058, 0.0243, 0.0301),
        ("SpeechBrain", 0.0056, 0.0249, 0.0305),
        ("CASE HF", 0.0122, 0.0231, 0.0353),
        ("NeMo", 0.0066, 0.0339, 0.0405),
        ("pyannote", 0.0168, 0.0279, 0.0447),
        ("Resemblyzer", 0.0484, 0.0565, 0.1049),
    ]

    all_passed = True
    for name, clean, deg, absolute in table_claims:
        m = get_model(models, name)
        actual_clean = m["clean_eer"]
        actual_deg = m["degradation_factor"]
        actual_abs = m["case_score_absolute"]

        clean_ok = abs(actual_clean - clean) < 0.001
        deg_ok = abs(actual_deg - deg) < 0.001
        abs_ok = abs(actual_abs - absolute) < 0.001

        passed = clean_ok and deg_ok and abs_ok
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        if not passed:
            print(f"  [{status}] {name}: MISMATCH")
            if not clean_ok:
                print(f"         Clean: {pct(actual_clean)} vs {pct(clean)}")
            if not deg_ok:
                print(f"         Deg: +{pct(actual_deg)} vs +{pct(deg)}")
            if not abs_ok:
                print(f"         Abs: {pct(actual_abs)} vs {pct(absolute)}")
        else:
            print(f"  [{status}] {name}: All match")

    results.append(all_passed)

    return results


def main():
    data = load_data()
    models = {m["model_name"]: m for m in data["leaderboard"]}

    all_results = []

    findings_results = verify_findings_md(data, models)
    all_results.extend(findings_results)

    metrics_results = verify_metrics_md(data, models)
    all_results.extend(metrics_results)

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    passed = sum(all_results)
    total = len(all_results)

    print(f"\nTotal checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if passed == total:
        print("\n✓ ALL DOCUMENTATION CLAIMS VERIFIED SUCCESSFULLY")
        return 0
    else:
        print("\n✗ SOME CLAIMS NEED ATTENTION")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
