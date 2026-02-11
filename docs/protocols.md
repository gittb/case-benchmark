# CASE Benchmark Protocols

The CASE Benchmark includes 24 evaluation protocols organized into 6 categories. Each protocol tests a specific carrier condition with clean enrollment audio.

## Protocol Naming Convention

```
clean_<category>_<variant>
```

- `clean_` prefix indicates clean enrollment
- `<category>` is one of: clean, codec, mic, noise, reverb, playback
- `<variant>` specifies the exact degradation applied

## Protocol List

### Clean (Baseline)

| Protocol | Description | Challenge |
|----------|-------------|-----------|
| `clean_clean` | Clean vs Clean | Baseline performance |

### Codec (7 protocols)

Tests robustness to audio codec compression artifacts.

| Protocol | Codec | Bitrate | Use Case |
|----------|-------|---------|----------|
| `clean_codec_gsm` | GSM-FR | 13 kbps | 2G mobile |
| `clean_codec_ulaw` | G.711 μ-law | 64 kbps | US telephony |
| `clean_codec_alaw` | G.711 A-law | 64 kbps | EU telephony |
| `clean_codec_opus_6k` | Opus | 6 kbps | Low-BW VoIP |
| `clean_codec_opus_12k` | Opus | 12 kbps | Standard VoIP |
| `clean_codec_opus_24k` | Opus | 24 kbps | HD VoIP |
| `clean_codec_mp3_32k` | MP3 | 32 kbps | Compressed audio |

**Expected difficulty**: GSM and low-bitrate Opus are hardest due to aggressive compression.

### Microphone (7 protocols)

Tests robustness to microphone frequency response variations.

| Protocol | Microphone Profile | Characteristics |
|----------|-------------------|-----------------|
| `clean_mic_webcam_budget` | Budget webcam | Bass rolloff, harsh highs |
| `clean_mic_webcam_quality` | Quality webcam | Flatter response |
| `clean_mic_headset_usb` | USB headset | Narrow band |
| `clean_mic_laptop_internal` | Laptop mic | Resonant, colored |
| `clean_mic_phone` | Phone mic | Telephony band |
| `clean_mic_smartphone_flagship` | Flagship phone | Wide band |
| `clean_mic_conference_ceiling` | Ceiling mic | Room coloration |

**Expected difficulty**: Budget webcam and laptop internal are hardest.

### Noise (5 protocols)

Tests robustness to additive background noise at various SNR levels.

| Protocol | SNR | Difficulty |
|----------|-----|------------|
| `clean_noise_snr_25` | 25 dB | Easy |
| `clean_noise_snr_20` | 20 dB | Moderate |
| `clean_noise_snr_15` | 15 dB | Challenging |
| `clean_noise_snr_10` | 10 dB | Hard |
| `clean_noise_snr_5` | 5 dB | Very hard |

**Noise types**: [DEMAND corpus](https://zenodo.org/record/1227121) (domestic, office, transport, nature, etc.)

> Note: The benchmark uses DEMAND noise (not MUSAN). If you train with MUSAN, your training data is properly separated.

### Reverb (1 protocol)

Tests robustness to room reverberation using **real Room Impulse Responses**.

| Protocol | Description |
|----------|-------------|
| `clean_reverb` | Real RIRs from OpenSLR-28 + BUT ReverbDB |

**RIR sources**: [OpenSLR-28](https://www.openslr.org/28/) (~417 real RIRs) + [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) (~1,500 real RIRs)

**Expected difficulty**: High - reverb smears temporal features.

> Note: The benchmark uses real RIRs (not simulated). If you train with pyroomacoustics or OpenSLR-26, your training data is properly separated.

### Playback (3 protocols)

Tests robustness to full playback chains (the hardest category).

| Protocol | Chain |
|----------|-------|
| `clean_playback_alaw_snr25_phone` | A-law → speaker → room → phone mic (SNR 25dB) |
| `clean_playback_gsm_snr20_webcam_budget` | GSM → speaker → room → budget webcam (SNR 20dB) |
| `clean_playback_ulaw_snr15_laptop_internal` | μ-law → speaker → room → laptop mic (SNR 15dB) |

**Expected difficulty**: Very high - combines codec, room acoustics, and microphone effects.

## Difficulty Ranking

Based on typical model performance:

1. **Easy**: clean_clean, clean_noise_snr_25, clean_mic_smartphone_flagship
2. **Moderate**: clean_codec_opus_24k, clean_noise_snr_20, clean_mic_headset_usb
3. **Challenging**: clean_codec_ulaw, clean_codec_alaw, clean_noise_snr_10
4. **Hard**: clean_codec_gsm, clean_reverb, clean_codec_opus_6k
5. **Very Hard**: All playback protocols

## Using Specific Protocols

You can evaluate on specific protocols instead of the full benchmark:

```python
from case_benchmark import CASEBenchmark, load_model

benchmark = CASEBenchmark("/path/to/benchmark")
model = load_model("speechbrain")

# Evaluate only codec protocols
codec_protocols = [p for p in benchmark.list_protocols() if "codec" in p]
results = benchmark.evaluate(model, protocols=codec_protocols)
```

Or via CLI:

```bash
case-benchmark evaluate \
    --model speechbrain \
    --benchmark-dir /path/to/benchmark \
    --protocols clean_clean clean_codec_gsm clean_reverb
```
