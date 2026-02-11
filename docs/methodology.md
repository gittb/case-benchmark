# CASE Benchmark Methodology

## Motivation

Current speaker verification benchmarks (VoxCeleb, SITW, CN-Celeb) evaluate models on audio with similar recording conditions to training data. Real-world deployments face a different challenge: **the same speaker may be encountered through vastly different acoustic carriers** - a phone call one day, a video conference the next, a recorded message played through laptop speakers later.

The CASE Benchmark directly measures this capability: **Can your model identify the same speaker regardless of how their voice was transmitted?**

## Core Design Principle

For each speaker in the evaluation set, we create multiple versions of their utterances:
- **Clean** (original high-quality)
- **Codec-degraded** (GSM, G.711, Opus, etc.)
- **Mic-filtered** (webcam, headset, laptop mic simulations)
- **Noise-added** (various SNR levels)
- **Reverb-added** (simulated room acoustics)
- **Playback chain** (codec → speaker → room → microphone)

Trials are constructed to test **cross-carrier matching**: enrollment in clean condition, verification in degraded condition.

## Data Source

The CASE Benchmark is built on two datasets:

**VoxCeleb1-O** (official test set):
- 40 speakers, ~400 utterances
- Celebrity interviews with varied recording conditions

**LibriSpeech test-clean**:
- 40 speakers, ~392 utterances
- Read audiobook speech in quiet conditions

This combination provides diversity in speaking styles (spontaneous vs read speech) and original recording conditions. Using standard test sets ensures the benchmark tests carrier robustness on speakers that may or may not have been seen during training, providing a fair evaluation.

## Degradation Categories

### 1. Codec Degradations

Audio codecs compress speech for transmission, introducing artifacts and frequency limitations.

| Codec | Bitrate | Use Case |
|-------|---------|----------|
| GSM-FR | 13 kbps | 2G mobile |
| G.711 μ-law | 64 kbps | North American PSTN |
| G.711 A-law | 64 kbps | European PSTN |
| Opus 6k | 6 kbps | Low-bandwidth VoIP |
| Opus 12k | 12 kbps | Standard VoIP |
| Opus 24k | 24 kbps | High-quality VoIP |
| MP3 32k | 32 kbps | Compressed storage |

### 2. Microphone Simulations

Consumer microphones have varying frequency responses that color the recorded audio.

| Profile | Characteristics |
|---------|-----------------|
| Webcam (budget) | Bass rolloff <200Hz, harsh HF |
| Webcam (quality) | Flatter response |
| USB Headset | Narrow band, speech-optimized |
| Laptop internal | Heavy bass rolloff, resonance |
| Phone | Telephony band (300-3400 Hz) |
| Smartphone (flagship) | Wide band, good response |
| Conference (ceiling) | Room coloration |

### 3. Noise Conditions

Additive background noise at various signal-to-noise (SNR) ratios.

| SNR | Difficulty |
|-----|------------|
| 25 dB | Easy (quiet background) |
| 20 dB | Moderate |
| 15 dB | Challenging |
| 10 dB | Difficult |
| 5 dB | Very difficult |

**Noise Source**: [DEMAND corpus](https://zenodo.org/record/1227121) (real environmental recordings: domestic, office, transport, etc.)

> **Data Leakage Warning**: The benchmark uses DEMAND noise, NOT MUSAN. If you train with MUSAN noise augmentation, this provides proper train/eval separation. If you train with DEMAND, there will be overlap - consider using MUSAN for training instead.

### 4. Reverb

Room reverberation applied using real **Room Impulse Responses (RIRs)**.

**RIR Sources (for benchmark)**:
- [OpenSLR-28](https://www.openslr.org/28/): ~417 real RIRs from various rooms
- [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database): ~1,500 real RIRs from 8 rooms with multiple microphone positions

> **Data Leakage Warning**: The benchmark uses **real RIRs** from OpenSLR-28 and BUT ReverbDB. To ensure proper train/eval separation:
> - **Recommended for training**: Simulated RIRs using [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) or [OpenSLR-26](https://www.openslr.org/26/)
> - **Avoid for training**: OpenSLR-28 and BUT ReverbDB (these are used in the benchmark)

This design ensures models are tested on **real acoustic conditions** they haven't seen during training.

### 5. Playback Chain (Most Challenging)

Full replay simulation: audio played through a speaker and re-recorded:

1. **Codec** applied (GSM, G.711)
2. **Speaker** simulation (laptop, phone speaker)
3. **Room acoustics** (reverb + early reflections)
4. **Microphone** recording (budget webcam, laptop mic)
5. **Noise** added (SNR 15-25 dB)

This represents the hardest real-world scenario: a voice message played back and recorded.

## Trial Generation

### Trial Counts
- **10,000 trials per protocol** (5,000 target + 5,000 impostor)
- **24 protocols** = 240,000 total trials
- **Balanced**: Equal same-speaker and different-speaker pairs

### Sampling Strategy

**Target trials** (same speaker):
- Randomly sample enrollment-verification pairs from same speaker
- Enrollment always clean, verification always degraded

**Impostor trials** (different speaker):
- Same-gender constraint to increase difficulty
- Random pairing across speakers

### Reproducibility

All random operations use fixed seeds:
- Trial generation: seed 42
- Degradation application: documented per-file

## Evaluation Metrics

### Primary: Equal Error Rate (EER)

The threshold where False Acceptance Rate = False Rejection Rate.

```
EER = threshold where FAR(t) = FRR(t)
```

Lower is better. Reported as percentage.

### Secondary: Minimum Detection Cost Function (minDCF)

NIST SRE standard metric with:
- P_target = 0.01
- C_miss = 1
- C_fa = 1

### Aggregate: Degradation Factor

The primary robustness metric - how much performance is lost due to carrier effects:

```
Degradation Factor = Absolute EER − Clean EER
```

Where:
- `Absolute EER` = weighted average EER across all protocols
- `Clean EER` = EER on clean_clean protocol (baseline)

**Interpretation**:
- **Lower is better** (more robust to carrier effects)
- A model with +2% degradation loses 2 percentage points due to carriers
- Independent of baseline performance - directly measures carrier susceptibility

> **Note on CASE-Score v1:** An earlier metric used normalized ratios (EER_degraded / EER_clean), but this can misleadingly reward models with poor baselines. See [Metrics](metrics.md) for full discussion.

## Limitations

1. **Simulated degradations**: Real codecs and microphones may differ from simulations
2. **VoxCeleb source**: Original recordings already have some channel effects
3. **English-centric**: VoxCeleb speakers are primarily English-speaking
4. **Single-condition trials**: Each trial tests one degradation (not combined)

## Citation

If you use the CASE Benchmark, please cite:

```bibtex
@misc{case_benchmark_2026,
  title={CASE Benchmark: Carrier-Agnostic Speaker Verification Evaluation},
  author={Gitter, Ben},
  year={2026},
  howpublished={\url{https://github.com/gittb/case-benchmark}}
}
```
