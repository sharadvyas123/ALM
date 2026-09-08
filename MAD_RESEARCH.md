You are right: we should not present it as if it appeared from nowhere.

Also, the name is **not “Time Diffusion Block.”** The proposed name was:

> **Multi-Scale Time–Frequency Fusion Block (MTFF Block)**

It is a proposed combination of ideas from existing research. That is normal in a research paper: you study existing methods, identify a gap, then design and test your own variation.

## Research basis

### 1. AST — your baseline

AST splits a log-Mel spectrogram into patches and sends them to a Transformer. It does not explicitly create separate local time and frequency feature branches before the Transformer. [AST: Audio Spectrogram Transformer — Gong et al.](https://arxiv.org/abs/2104.01778)

```text
Spectrogram → patches → Transformer → prediction
```

Your work begins from this baseline.

### 2. SepTr — separate time and frequency processing

SepTr argues that time and frequency are different axes and should not necessarily be processed in the same way. It applies attention along one axis and then the other. This is the closest theoretical basis for our time branch and frequency branch idea. [SepTr: Separable Transformer for Audio Spectrogram Processing](https://arxiv.org/abs/2203.09581)

```text
Spectrogram
    ↓
Frequency-aware processing
    ↓
Time-aware processing
```

Your difference is that you keep the AST architecture and introduce a lightweight fusion block before the AST encoder.

### 3. SpecTNT — spectral features plus temporal features

SpecTNT explicitly models frequency-related patterns and then temporal patterns. It supports the idea that audio Transformers benefit when spectral and temporal information are handled separately. [SpecTNT: a Time-Frequency Transformer for Music Audio](https://arxiv.org/abs/2110.09127)

### 4. Multi-time-scale convolution

Your multiple temporal kernels—short, medium, and long temporal context—are based on multi-time-scale feature extraction. This helps models recognize patterns with different durations. [Multi-Time-Scale Convolution for Speech Audio Signals](https://arxiv.org/abs/2003.03375)

```text
Small time kernel  → sudden Gunshot
Medium time kernel → Footsteps
Large/dilated kernel → Vehicle or Helicopter
```

### 5. Directly related time–frequency multi-scale work

There is already research using parallel time–frequency multi-scale attention with convolution for environmental-sound classification. This means we must cite it and ensure our block is not simply a copy. [Parallel Time-Frequency Multi-Scale Attention with Dynamic Convolution](https://doi.org/10.3390/e27101007)

---

## Your actual research contribution

Do not claim:

> “We invented time–frequency processing for audio.”

That would be incorrect.

Instead, claim something like:

> “Inspired by separable time–frequency modeling and multi-scale audio feature extraction, we propose a Multi-Scale Time–Frequency Fusion block for AST-based multi-label military sound classification.”

Your specific proposed design is:

```text
AST patch embeddings
        ↓
Reshape tokens into time–frequency patch grid
        ↓
Parallel temporal multi-scale convolutions
        +
Parallel frequency multi-scale convolutions
        ↓
Learnable gate fuses both branches
        ↓
Enhanced AST tokens
        ↓
Original AST Transformer
```

The possible novelty is the combination of:

```text
1. AST backbone
2. Parallel multi-scale time branch
3. Parallel multi-scale frequency branch
4. Learnable gated fusion
5. Multi-label military audio mixtures
6. Optional class-query output head
```

But novelty is not proven just by designing the block. You prove its value with experiments:

```text
AST baseline
AST + temporal branch only
AST + frequency branch only
AST + both branches without gate
AST + full gated MTFF block
```

If the full model gives better mAP, Macro F1, and per-class AP—especially for Gunshot, Vehicle, Helicopter, and Shelling—you can demonstrate that the proposed block contributes something meaningful.