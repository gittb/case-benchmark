# Why This Benchmark Exists

## The Bigger Picture

I'm interested in a problem that most people don't think about: **human memory is lossy**.

We remember *what* decisions we made, but not *why*. Three years later, you know you chose Option A, but the context - the discussions, the tradeoffs, the reasoning - is gone. All that remains is the choice itself. And when a similar decision comes up, you're left asking: *Do I trust my past self?*

I think a lot about systems that could help with this. Imagine being able to index your life's conversations - not just the typed ones, but the *talked* ones. The discussions with colleagues, the phone calls with friends, the brainstorming sessions that happened in a room somewhere. What if we could autonomously extract structure from unstructured audio: identify speakers, extract topics, create a searchable index of events?

This is technically possible today. LLMs can transcribe, summarize, and extract. Speaker diarization can identify *who* is talking. The pieces exist.

But there's a fundamental problem.

## The Pain Point

My friends appear in my life through multiple channels. I talk to them over VoIP (Discord, Zoom). I call them on the phone. I spend time with them in person.

When I tried to build a system that correlates speakers across these sources, I hit a wall: **the same person produces different embeddings depending on how I encountered them**.

The speaker verification models that achieve <1% error on clean benchmarks degrade dramatically when audio passes through:
- Phone codecs (GSM, G.711)
- VoIP compression (Opus)
- Laptop microphones
- Room acoustics
- The full chain: codec → speaker → room → microphone

We're talking up to 19x worse performance. A model that's 99.5% accurate on clean audio might be 90% accurate on a Zoom call - and that's not good enough for autonomous indexing.

## The Frustrating Part

Humans don't have this problem. We recognize voices regardless of the medium. We can identify a friend whether they're standing next to us or calling from a noisy airport on a bad connection. Our auditory system is *carrier-agnostic* in a way that current models are not.

## The Approach

I had two options:
1. **Engineer around it**: Build separate voice profiles for each carrier, maintain complex matching logic, accept the fragility
2. **Attack the root cause**: Make speaker embeddings robust to carrier effects

I chose the second path. But you can't improve what you can't measure.

## Why Measurement Matters

I'm not the best model trainer in the world. But I am good at architecting evaluation. And the fundamental truth is: **you can't make progress without being able to measure progress**.

The existing benchmarks (VoxCeleb, SITW, CN-Celeb) don't test carrier robustness. They tell you how good a model is on clean audio, which is useful - but it doesn't predict how that model will perform when your friend calls you from their car.

So I built the CASE Benchmark to lift the fog of war on carrier robustness. It tests models across:
- **Codecs**: The compression your voice goes through on phone calls and VoIP
- **Microphones**: The frequency response of webcams, laptops, phones
- **Noise**: Background sound at various levels
- **Reverb**: Room acoustics from real recorded impulse responses
- **Playback chains**: The full codec→speaker→room→mic pipeline

These aren't exotic conditions. They're the conditions we encounter every day in remote work, virtual interactions, and modern communication.

## The Model

I also trained a model (CASE HF v2-512) that's slightly more robust. It's not groundbreaking - it doesn't have the best clean performance - but it has the **lowest degradation factor** among tested models. It holds up better when conditions aren't ideal.

That's the tradeoff I was optimizing for. Not peak performance, but consistent performance.

## The Hope

This benchmark is a jumping-off point. I hope it gives the audio ML community a tool to:
- Measure carrier robustness alongside clean performance
- Identify where models fail in realistic conditions
- Develop training approaches that produce carrier-agnostic embeddings

The goal isn't just better benchmarks. It's speaker embeddings that work like human hearing - recognizing the same person regardless of how their voice reaches us.

That's the future I'm working toward. This is one step.

---

* -  Ben Gitter, 2026*
