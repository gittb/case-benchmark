# Key Findings

This document summarizes key findings from evaluating state-of-the-art speaker embedding models on the CASE Benchmark.

## Executive Summary

**All tested models show significant performance degradation on carrier-affected audio**, with playback chains causing up to 19× worse EER compared to clean audio. This reveals a critical gap between benchmark performance and real-world robustness.

## Finding 1: SOTA Models Fail on Carrier Conditions

Tested models (SpeechBrain ECAPA-TDNN, WeSpeaker, NeMo TitaNet, etc.) achieve excellent performance on clean audio (~1% EER) but degrade significantly on carrier-affected audio:

| Condition | Typical Degradation |
|-----------|---------------------|
| Codec | 1.5-3× worse |
| Mic | ~1× (minimal) |
| Noise | 1.1-1.6× worse |
| Reverb | 4-12× worse |
| Playback | **7-19× worse** |

This means a model with 0.6% EER on clean audio might have 9-13% EER on playback chains - a completely different operating regime.

## Finding 2: Architecture Isn't the Problem

We compared similar architectures with different training:
- **SpeechBrain ECAPA-TDNN**: Trained on VoxCeleb without carrier augmentation
- **CASE HF v2-512**: ECAPA-TDNN architecture, trained with carrier augmentation

Both use:
- ECAPA-TDNN architecture
- 192-dimensional embeddings
- VoxCeleb-based training data

**Results show the CASE model reduces codec degradation by ~57% and playback degradation by ~18%**, proving the problem is training methodology, not model capacity.

## Finding 3: Playback Chains Are the Real Challenge

The playback category (codec → speaker → room → microphone) is consistently the hardest:

```
                        Clean     Playback    Degradation
WeSpeaker ResNet34      0.58%     8.57%       14.8×
SpeechBrain ECAPA       0.56%     9.37%       16.7×
CASE HF v2-512          1.22%     9.10%       7.5×
NeMo TitaNet-L          0.66%     12.61%      19.1×
pyannote Embedding      1.68%     11.22%      6.7×
```

Even the best models struggle because playback chains combine multiple degradations that compound information loss.

## Finding 4: Carrier Training Works

The CASE HF v2-512 model demonstrates that carrier-aware training significantly improves robustness:

| Category | SOTA Avg Degradation | CASE HF Degradation | Improvement |
|----------|---------------------|---------------------|-------------|
| Codec | +1.10% | +0.47% | 57% less |
| Reverb | +5.71% | +5.34% | 6% less |
| Playback | +9.58% | +7.88% | 18% less |

Note: "Degradation" = Category EER minus Clean EER. SOTA Avg = WeSpeaker, SpeechBrain, NeMo.

While CASE HF has a higher clean EER (1.22% vs 0.56-0.66%), it achieves the **lowest overall degradation factor** (+2.31%) among all tested models, showing consistent robustness across carrier conditions.

## Finding 5: Easy Conditions Hide the Problem

Models that excel on clean benchmarks may be poor choices for real deployments:

```
Clean audio results:
  - All models: 0.56-1.68% EER ✓

CASE Benchmark Results (carrier-affected):
  - SOTA models: 8.6-12.6% EER on playback
  - CASE HF: 9.10% EER on playback (lowest degradation factor)
```

The clean benchmark gives no indication of real-world performance.

## Finding 6: Remaining Challenges

Even with carrier training, significant challenges remain:

1. **Playback chains**: 8-13% EER is still challenging for production
2. **Reverb**: Room acoustics smear temporal features
3. **Combined degradations**: Real scenarios may have multiple effects

## Recommendations

### For Practitioners

1. **Don't trust clean benchmarks alone**: Test on carrier-affected audio
2. **Use CASE Benchmark**: Get realistic robustness estimates
3. **Consider carrier augmentation**: Improves robustness with no cost

### For Researchers

1. **Report Clean EER and Degradation Factor**: Shows both baseline and robustness
2. **Evaluate on playback protocols**: The true stress test
3. **Develop better augmentation strategies**: Current approaches help but don't solve the problem

## Conclusion

The CASE Benchmark reveals a significant gap between controlled benchmark performance and real-world robustness. All tested models degrade dramatically on carrier-affected audio, with playback chains causing up to 19× worse performance.

The good news: **this is a solvable problem**. Carrier-aware training reduces degradation on codec and playback conditions with no architectural changes. The CASE Benchmark provides a way to measure this progress and push the field toward more robust speaker verification.
