# CASE Benchmark Metrics

## Overview

The CASE Benchmark measures **how susceptible a speaker verification model is to performance degradation across carrier conditions**. It tests models on codecs, microphones, noise, reverb, and playback chains - not as a weighted simulation of real-world frequency, but as a comprehensive stress test of robustness.

This document explains the metrics used to evaluate and compare models.

## Primary Metrics

These two metrics together tell the complete story of a model's carrier robustness.

### Clean EER

The Equal Error Rate on clean (unprocessed) audio. This establishes the baseline - how good is the model under ideal conditions?

```
Clean EER = EER on clean_clean protocol
```

**Interpretation:**
- **Lower is better**
- Typical SOTA models achieve 0.5-1.5% on VoxCeleb1-O
- This is your starting point before carrier effects

### Degradation Factor

The absolute increase in error rate when moving from clean to carrier-affected audio. This is the core measure of carrier robustness.

```
Degradation Factor = Absolute EER − Clean EER
```

**Interpretation:**
- **Lower is better** (more robust to carrier effects)
- A model with +2% degradation loses 2 percentage points of accuracy due to carriers
- Independent of baseline performance - directly measures robustness

**Example:**
| Model | Clean EER | Absolute EER | Degradation Factor |
|-------|-----------|--------------|-------------------|
| **CASE HF v2-512 (My Model)** | 1.22% | 3.53% | **+2.31%** |
| WeSpeaker ResNet34 | 0.58% | 3.01% | **+2.43%** |


CASE HF has the lower degradation factor (+2.31% vs +2.43%), meaning it's more robust to carrier effects, even though WeSpeaker has better absolute performance.

## Supporting Metrics

### Absolute EER

The weighted average EER across all protocols. Useful for understanding overall performance but depends on which conditions you encounter in deployment.

```
Absolute EER = weighted_avg(EER across all 24 protocols)
```

**Interpretation:**
- **Lower is better**
- Represents average performance if you encountered all carrier conditions equally
- Not weighted for real-world deployment frequency (that varies by application)

### Category Breakdown

Average EER per degradation category. This diagnostic tool identifies where a model struggles.

| Category | Description | # of Protocols |
|----------|-------------|-------------|
| Clean | Baseline (no degradation) | 1 |
| Codec | GSM-FR (2G mobile), G.711 A-law (European PSTN), G.711 μ-law (North American PSTN), Opus (6k/12k/24k VoIP/WebRTC), MP3 (32k streaming) | 7 |
| Mic | Budget webcam, quality webcam, USB headset, laptop internal, phone handset, flagship smartphone, conference ceiling | 7 |
| Noise | Additive noise (DEMAND corpus) at SNR 5, 10, 15, 20, 25 dB | 5 |
| Reverb | Real room impulse responses (OpenSLR-28 + BUT ReverbDB) | 1 |
| Playback | Full codec→speaker→room→mic chain (A-law + phone, GSM + webcam, μ-law + laptop) | 3 |

During benchmark sample generation, trial pairs (enrollment and test utterances) are **randomly sampled** with a fixed seed for reproducibility. Speaker and utterance selections, as well as RIR and noise file assignments, are all drawn randomly and pre-computed in a deterministic specification to ensure byte-for-byte reproducibility across runs. Each protocol within a category contains an **equal number of trials** (10,000 per protocol: 5,000 target pairs + 5,000 impostor pairs), ensuring uniform representation across all degradation conditions.

**Example (WeSpeaker ResNet34):**
| Category | Avg EER | vs Clean |
|----------|---------|----------|
| Clean | 0.58% | baseline |
| Codec | 1.73% | +1.15% |
| Mic | 0.59% | +0.01% |
| Noise | 0.73% | +0.15% |
| Reverb | 5.88% | +5.30% |
| Playback | 8.57% | +7.99% |

This reveals that playback chains and reverb cause the most degradation for this model.

## CASE-Score v1 (Normalized Ratio)  -  Use With Caution

CASE-Score v1 measures relative degradation as a ratio of degraded EER to clean EER:

```
CASE-Score v1 = avg(EER_category / EER_clean) across categories
```

**The Problem:** This metric can be misleading because it rewards models with poor baseline performance.

### Why the Normalized Ratio Fails

Consider Resemblyzer vs WeSpeaker:

| Model | Clean EER | Absolute EER | CASE-Score v1 | Degradation Factor |
|-------|-----------|--------------|---------------|-------------------|
| Resemblyzer | 4.84% | 10.49% | **2.17x** | +5.65% |
| WeSpeaker | 0.58% | 3.01% | 5.19x | **+2.43%** |

By CASE-Score v1, Resemblyzer looks more "robust" (2.17x vs 5.19x) because:
- It started with 4.84% EER (already poor)
- It degraded to 10.49% (only ~2x worse relatively)

But WeSpeaker is objectively better:
- It started with 0.58% EER (excellent)
- It degraded to 3.01% (still good in absolute terms)
- Its actual performance loss (+2.43%) is less than half of Resemblyzer's (+5.65%)

**The normalized ratio punishes models for being good at clean audio.** A model that starts excellent and degrades to "good" scores worse than a model that starts poor and degrades to "terrible."

### When CASE-Score v1 Is Useful

The ratio can be meaningful when comparing models with **similar clean EER**. If two models both achieve ~0.6% clean EER, then comparing their degradation ratios is valid. But across models with different baselines, use Degradation Factor instead.

## How to Compare Models

### Decision Framework

1. **Similar clean EER?** → Compare degradation factors directly
2. **Different clean EER?** → Consider the tradeoff:
   - Some models (like CASE HF) trade clean performance for robustness
   - Decide based on your deployment: Is baseline accuracy or robustness more critical?
3. **Know your deployment conditions?** → Look at category breakdown
   - VoIP application? Focus on codec performance
   - Smart speaker? Focus on playback chains

### Full Comparison Table

| Model | Clean EER | Degradation | Absolute EER | Best For |
|-------|-----------|-------------|--------------|----------|
| **CASE HF v2-512 (My Model)** | 1.22% | **+2.31%** | 3.53% | **Most robust to degradation** |
| WeSpeaker ResNet34 | **0.58%** | +2.43% | **3.01%** | Best overall performance |
| SpeechBrain ECAPA-TDNN | 0.56% | +2.49% | 3.05% | Strong baseline |
| NeMo TitaNet-L | 0.66% | +3.39% | 4.05% | Good clean, less robust |
| pyannote Embedding | 1.68% | +2.79% | 4.47% | Moderate all-around |
| Resemblyzer | 4.84% | +5.65% | 10.49% | Legacy/lightweight only |

## Equal Error Rate (EER)

### Definition

EER is the operating point where False Acceptance Rate equals False Rejection Rate:

```
FAR = False Accepts / Total Impostors
FRR = False Rejects / Total Targets
EER = threshold t where FAR(t) = FRR(t)
```

### Interpretation

- **Lower is better**
- Reported as a percentage (e.g., 1.5% EER)
- Standard metric in speaker verification research
- Does not require choosing a specific threshold

### Typical Values

| Performance | EER Range |
|-------------|-----------|
| Excellent | < 1% |
| Good | 1-3% |
| Acceptable | 3-5% |
| Poor | 5-10% |
| Very Poor | > 10% |

## Minimum Detection Cost Function (minDCF)

### Definition

minDCF is the NIST Speaker Recognition Evaluation (SRE) standard metric that accounts for prior probabilities and costs:

```
DCF(t) = C_miss × P_miss(t) × P_target + C_fa × P_fa(t) × (1 - P_target)
minDCF = min_t DCF(t) / min(C_miss × P_target, C_fa × (1 - P_target))
```

### Parameters

The CASE Benchmark uses standard NIST parameters:
- P_target = 0.01 (1% prior for target speakers)
- C_miss = 1 (cost of missing a target)
- C_fa = 1 (cost of false alarm)

### Interpretation

- **Lower is better**
- Range: 0 (perfect) to 1 (random chance)
- More stable than EER for imbalanced test sets
- Reflects application-specific costs

## Computing Metrics Programmatically

```python
from case_benchmark.metrics import compute_eer, compute_min_dcf, compute_degradation

# Per-trial scores and labels
scores = model_scores  # numpy array
labels = trial_labels  # numpy array (1=target, 0=impostor)

# Compute EER
eer, threshold = compute_eer(scores, labels)
print(f"EER: {eer * 100:.2f}%")

# Compute minDCF
min_dcf, dcf_threshold = compute_min_dcf(scores, labels)
print(f"minDCF: {min_dcf:.4f}")

# Compute degradation metrics from results
from case_benchmark.metrics import compute_benchmark_metrics

results = benchmark.evaluate(model)
metrics = compute_benchmark_metrics(results)

print(f"Clean EER: {metrics.clean_eer * 100:.2f}%")
print(f"Absolute EER: {metrics.absolute_eer * 100:.2f}%")
print(f"Degradation Factor: +{metrics.degradation_factor * 100:.2f}%")
```
