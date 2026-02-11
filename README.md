# CASE Benchmark

**Carrier-Agnostic Speaker Verification Evaluation**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE-CODE)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

> **Why this exists:** I wanted to build a system that indexes spoken conversations - automatically identifying speakers across Discord, phone calls, and in-person meetings. But I hit a wall: the same person produces different embeddings depending on how I encountered them. Current models degrade up to 19x on real-world audio. Humans don't have this problem - we recognize voices regardless of medium. This benchmark measures that gap. [Read the full story →](docs/motivation.md)

> **Companion model:** Alongside the benchmark, we release [CASE HF v2-512](https://huggingface.co/bigstorm/case-speaker-embedding-v2-512), a carrier-augmented model that demonstrates training methodology alone can reduce carrier degradation. It uses the same ECAPA-TDNN architecture as the SpeechBrain baseline but achieves the lowest degradation factor on the leaderboard, showing that this gap is actionable, not just measurable.

---

## Can Your Model Handle Real-World Audio?

State-of-the-art speaker verification models achieve <1% EER on clean benchmarks.
**But what happens when audio passes through phone codecs, cheap webcams, or is replayed through speakers?**

The CASE Benchmark answers this question.

### The Problem

| Condition | Typical SOTA Performance |
|-----------|--------------------------|
| Clean Audio | **0.6-1.7% EER** |
| Phone Codec | 2-4% EER |
| Laptop Microphone | 0.6-1.8% EER |
| Room Reverb | 5-8% EER |
| **Playback Chain** | **9-13% EER** |

**That's up to 19x worse performance** on realistic conditions.

### What is a "Playback Chain"?

The hardest scenario: audio encoded, played through a speaker, and re-recorded:

```
Voice → [Codec] → [Speaker] → [Room Acoustics] → [Microphone] → Recording
```

This happens when:
- Voice messages are played back and recorded
- Conference calls with speaker playback
- Smart speaker interactions
- Any voice replay attack scenario

---

## Quick Start

### Installation

```bash
pip install case-benchmark

# Install with model support
pip install case-benchmark[speechbrain]  # SpeechBrain ECAPA-TDNN
pip install case-benchmark[all-models]   # All supported models
```

### Download Benchmark Data

```bash
case-benchmark download --output-dir ./benchmark_data
```

### Evaluate Your Model

```bash
# Using built-in model wrappers
case-benchmark evaluate \
    --model speechbrain \
    --benchmark-dir ./benchmark_data \
    --output-dir ./results

# Or programmatically
```

```python
from case_benchmark import CASEBenchmark, load_model

benchmark = CASEBenchmark("./benchmark_data")
model = load_model("speechbrain")

results = benchmark.evaluate(model)
results.print_summary()
# Clean EER: 0.56%, Degradation: +2.49%
```

---

## Benchmark Results

### Leaderboard

| Rank | Model | Absolute EER | Degradation | Clean EER |
|------|-------|-------------|-------------|-----------|
| 1 | **CASE HF v2-512 (My Model)** | 3.53% | **+2.31%** | 1.22% |
| 2 | WeSpeaker ResNet34 | **3.01%** | +2.43% | 0.58% |
| 3 | SpeechBrain ECAPA-TDNN | 3.05% | +2.49% | 0.56% |
| 4 | pyannote Embedding | 4.47% | +2.79% | 1.68% |
| 5 | NeMo TitaNet-L | 4.05% | +3.39% | 0.66% |
| 6 | Resemblyzer | 10.49% | +5.65% | 4.84% |

**Key Finding**: The CASE HF model achieves the **lowest degradation factor** (+2.31%), validating its carrier-agnostic design.

### Context: VoxCeleb1-O SOTA

For reference, current SOTA on VoxCeleb1-O (clean-clean only):
- **ResNet293 + VoxBlink2**: 0.17% EER ([arXiv:2407.11510](https://arxiv.org/abs/2407.11510))
- **ERes2NetV2**: 0.61% EER ([3D-Speaker](https://github.com/modelscope/3D-Speaker))

Our benchmark tests **production-ready models** that are easily accessible. The models above require specialized training (VoxBlink2 dataset, 100K+ speakers) and post-processing (AS-Norm, QMF) not typically used in deployment.

### Category Breakdown (WeSpeaker ResNet34)

| Category | Avg EER | vs Clean |
|----------|---------|----------|
| Clean | 0.58% | baseline |
| Codec | 1.73% | +1.15% |
| Mic | 0.59% | +0.01% |
| Noise | 0.73% | +0.15% |
| Reverb | 5.88% | +5.30% |
| Playback | 8.57% | +7.99% |

### Key Insight: Playback Chains Remain Challenging

All models show significant degradation on playback scenarios (codec→speaker→room→mic chains), though carrier-aware training reduces this gap substantially.

---

## Evaluation Protocols

The benchmark includes **24 protocols** across 6 categories:

| Category | Protocols | Description |
|----------|-----------|-------------|
| **Clean** | 1 | Baseline (clean vs clean) |
| **Codec** | 7 | GSM, G.711, Opus, MP3 |
| **Mic** | 7 | Webcam, laptop, phone, headset |
| **Noise** | 5 | SNR 5-25 dB |
| **Reverb** | 1 | Simulated room acoustics |
| **Playback** | 3 | Full codec→speaker→room→mic chain |

Each protocol has 10,000 trials (5,000 target + 5,000 impostor).

---

## Metrics

Two metrics together describe a model's carrier robustness:

### 1. Clean EER (Baseline)
```
Clean EER = EER on clean_clean protocol
```
- Measures **baseline performance** under ideal conditions
- **Lower is better** (e.g., 0.58% is excellent)

### 2. Degradation Factor (Robustness)
```
Degradation = Absolute EER − Clean EER
```
- Measures **robustness**: how much performance is *lost* due to carrier effects
- **Lower is better** (e.g., +2.31% means minimal degradation)
- Independent of baseline - directly measures carrier susceptibility

A model with low Clean EER and low Degradation is ideal. Some models (like CASE HF) trade baseline performance for better robustness.

> **Note:** An earlier "CASE-Score v1" metric used normalized ratios (EER_degraded / EER_clean), but this can misleadingly reward models with poor baselines. See [Metrics](docs/metrics.md) for full details.

---

## Supported Models

Built-in wrappers for popular models:

| Model | Install | Status |
|-------|---------|--------|
| SpeechBrain ECAPA-TDNN | `pip install case-benchmark[speechbrain]` | ✅ Supported |
| WeSpeaker ResNet34/CAM++ | `pip install case-benchmark[wespeaker]` | ✅ Supported |
| pyannote embedding | `pip install case-benchmark[pyannote]` | ✅ Supported |
| NVIDIA NeMo TitaNet | `pip install case-benchmark[nemo]` | ✅ Supported |
| Resemblyzer | `pip install case-benchmark[resemblyzer]` | ✅ Supported |
| CASE HF v2-512 | `pip install case-benchmark[case-hf]` | ✅ Supported |

### Custom Models

Implement the `EmbeddingModel` interface:

```python
from case_benchmark.models.base import EmbeddingModel
import numpy as np
from pathlib import Path

class MyModel(EmbeddingModel):
    def load(self, device: str = "cpu") -> None:
        self.model = load_my_model(device)
        self._loaded = True

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        audio = load_audio(audio_path)
        return self.model.encode(audio).numpy()

    @property
    def embedding_dim(self) -> int:
        return 192

    @property
    def name(self) -> str:
        return "My Custom Model"
```

---

## Data

### Source
- **VoxCeleb1-O**: 40 speakers, ~400 utterances (official test set)
- **LibriSpeech test-clean**: 40 speakers, ~392 utterances
- **Total**: 80 speakers across both datasets
- **Sample rate**: 16kHz mono

### Degradations Applied
- **Codecs**: GSM, G.711 (μ-law, A-law), Opus (6k/12k/24k), MP3
- **Microphones**: Simulated FIR filters for webcam, laptop, phone, etc.
- **Noise**: [DEMAND corpus](https://zenodo.org/record/1227121) at various SNR levels
- **Reverb**: Real RIRs from [OpenSLR-28](https://www.openslr.org/28/) + [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database)
- **Playback**: Full codec→speaker→room→mic chain

### Avoiding Data Leakage

> **Important**: The benchmark uses **different data sources** than typical training pipelines to ensure proper train/eval separation.

| Component | Benchmark Source | Recommended for Training |
|-----------|------------------|--------------------------|
| **Noise** | [DEMAND](https://zenodo.org/record/1227121) | [MUSAN](https://www.openslr.org/17/) ✓ |
| **Reverb** | [OpenSLR-28](https://www.openslr.org/28/) + [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) (real RIRs) | [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) or [OpenSLR-26](https://www.openslr.org/26/) (simulated) ✓ |

If you train with MUSAN noise and pyroomacoustics/OpenSLR-26 RIRs, your training data is properly separated from the benchmark. See [docs/methodology.md](docs/methodology.md) for details.

### Download

```bash
# From HuggingFace
case-benchmark download --output-dir ./benchmark_data

# Or using huggingface_hub directly
from huggingface_hub import snapshot_download
snapshot_download("bigstorm/case-benchmark", local_dir="./benchmark_data")
```

---

## Documentation

- [Why This Exists](docs/motivation.md) - The problem this benchmark is trying to solve
- [Methodology](docs/methodology.md) - Benchmark design and technical approach
- [Protocols](docs/protocols.md) - Detailed protocol descriptions
- [Metrics](docs/metrics.md) - EER, Degradation Factor, and how to compare models
- [Findings](docs/findings.md) - Key results and analysis
- [Submission Guide](docs/submission.md) - How to submit to leaderboard

---

## Citation

If you use the CASE Benchmark in your research, please cite:

```bibtex
@misc{gitter2026case,
  title={CASE Benchmark: Carrier-Agnostic Speaker Verification Evaluation},
  author={Gitter, Ben},
  year={2026},
  howpublished={\url{https://github.com/gittb/case-benchmark}}
}
```

---

## License

- **Code**: [MIT License](LICENSE-CODE)
- **Data**: [CC BY-NC 4.0](LICENSE-DATA) (non-commercial research only) (Contact Ben Gitter for a commerical license)

The benchmark audio is derived from VoxCeleb and LibriSpeech, which have their own license terms.

---

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Report issues on [GitHub Issues](https://github.com/gittb/case-benchmark/issues)
- Submit model results via [Pull Request](docs/submission.md)

---

## Acknowledgments

- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) for source audio data (VoxCeleb1-O test set)
- [LibriSpeech](https://www.openslr.org/12/) for source audio data (test-clean subset)
- [DEMAND](https://zenodo.org/record/1227121) for noise samples used in the benchmark
- [OpenSLR-28](https://www.openslr.org/28/) and [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) for real room impulse responses
