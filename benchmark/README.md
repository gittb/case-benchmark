---
license: cc-by-nc-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - speaker-verification
  - speaker-recognition
  - audio
  - benchmark
pretty_name: CASE Benchmark
size_categories:
  - 10K<n<100K
---

# CASE Benchmark

**Carrier-Agnostic Speaker Embedding Benchmark** - A comprehensive benchmark for evaluating speaker verification systems under real-world acoustic carrier conditions.

## Overview

CASE Benchmark tests speaker embedding models across 24 protocols covering:
- **Clean**: Matched enrollment and test conditions
- **Codecs** (7): GSM, G.711 μ-law/A-law, Opus (6k/12k/24k), MP3
- **Microphones** (7): Laptop, webcam, phone, headset, conference
- **Noise** (5): SNR levels from 5dB to 25dB
- **Reverb** (1): Simulated room acoustics
- **Playback chains** (3): Combined codec + noise + microphone

## Quick Start

```python
from case_benchmark.download import download_benchmark

# Download full benchmark (~3.1GB compressed)
download_benchmark("./benchmark")

# Download only specific conditions
download_benchmark("./benchmark", conditions=["clean", "codec"])

# Download only VoxCeleb1-O dataset
download_benchmark("./benchmark", datasets=["voxceleb1_o"])
```

## Structure

Audio files are stored as compressed archives per condition:

```
audio/
├── voxceleb1_o_clean.tar.gz     # 69MB  - 400 files
├── voxceleb1_o_codec.tar.gz     # 464MB - 2,800 files
├── voxceleb1_o_mic.tar.gz       # 467MB - 2,800 files
├── voxceleb1_o_noise.tar.gz     # 346MB - 2,000 files
├── voxceleb1_o_reverb.tar.gz    # 71MB  - 400 files
├── voxceleb1_o_playback.tar.gz  # 205MB - 1,200 files
├── librispeech_clean.tar.gz     # 63MB  - 392 files
├── librispeech_codec.tar.gz     # 426MB - 2,744 files
├── librispeech_mic.tar.gz       # 430MB - 2,744 files
├── librispeech_noise.tar.gz     # 331MB - 1,960 files
├── librispeech_reverb.tar.gz    # 67MB  - 392 files
└── librispeech_playback.tar.gz  # 196MB - 1,176 files

trials/
├── clean_clean.txt
├── clean_codec_*.txt    # 7 codec protocols
├── clean_mic_*.txt      # 7 microphone protocols
├── clean_noise_*.txt    # 5 noise protocols
├── clean_reverb.txt
└── clean_playback_*.txt # 3 playback chain protocols
```

After extraction, audio files are organized as:
```
voxceleb1_o/
├── clean/{speaker_id}/utt_*.wav
├── codec/{codec_type}/{speaker_id}/utt_*.wav
├── mic/{mic_type}/{speaker_id}/utt_*.wav
├── noise/{snr_level}/{speaker_id}/utt_*.wav
├── reverb/{speaker_id}/utt_*.wav
└── playback/{chain_type}/{speaker_id}/utt_*.wav
```

## Trial Format

Each trial file contains lines in the format:
```
<label> <enrollment_path> <test_path>
```

Where:
- `label`: 1 for same speaker, 0 for different speaker
- `enrollment_path`: Path to enrollment audio (always clean)
- `test_path`: Path to test audio (condition-dependent)

## Datasets

| Dataset | Speakers | Utterances | Source |
|---------|----------|------------|--------|
| VoxCeleb1-O | 40 | 400 clean | VoxCeleb1 test set |
| LibriSpeech | 40 | 392 clean | LibriSpeech test-clean |

## License

- **Audio data**: CC BY-NC 4.0 (non-commercial use)
- **Evaluation code**: MIT License

## Citation

```bibtex
@misc{case-benchmark-2026,
  title={CASE Benchmark: Carrier-Agnostic Speaker Embedding Evaluation},
  author={Gitter, Ben},
  year={2026},
  url={https://github.com/gittb/case-benchmark}
}
```
