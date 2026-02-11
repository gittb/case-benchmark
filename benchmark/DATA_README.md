# CASE Benchmark Audio Data

This dataset contains audio files for the CASE (Carrier-Agnostic Speaker Embedding) Benchmark.

## Structure

```
benchmark/
├── voxceleb1_o/          # VoxCeleb1-O test set speakers
│   ├── clean/            # Original clean audio
│   ├── codec/            # Codec-processed (alaw, gsm, mp3, opus, ulaw)
│   ├── mic/              # Microphone-simulated
│   ├── noise/            # Noise-added (SNR 5-25dB)
│   ├── reverb/           # Reverb-added
│   └── playback/         # Full playback chains
├── librispeech/          # LibriSpeech test speakers (same structure)
└── trials/               # Trial files for evaluation
```

## Conditions

| Category | Conditions |
|----------|------------|
| **Codec** | A-law, μ-law, GSM, MP3 32k, Opus 6k/12k/24k |
| **Microphone** | Conference ceiling, headset USB, laptop internal, phone, smartphone, webcam |
| **Noise** | DEMAND noise at SNR 5/10/15/20/25 dB |
| **Reverb** | Real RIRs from OpenSLR-28 + BUT ReverbDB |
| **Playback** | Combined codec → speaker → room → mic chains |

## Usage

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download full benchmark
snapshot_download(
    repo_id="bigstorm/case-benchmark",
    repo_type="dataset",
    local_dir="./benchmark"
)

# Or download specific files
trials = hf_hub_download(
    repo_id="bigstorm/case-benchmark",
    repo_type="dataset",
    filename="trials/clean_clean.txt"
)
```

## License

Audio data: CC BY-NC 4.0 (Non-commercial use only)
Code: MIT License

## Citation

```bibtex
@misc{case-benchmark-2024,
  title={CASE Benchmark: Carrier-Agnostic Speaker Verification Evaluation},
  author={Gitter, Ben},
  year={2024},
  url={https://github.com/gittb/case-benchmark}
}
```
