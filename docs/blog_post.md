# How I Beat torchaudio at Spectrogram Generation with a Custom CUDA Kernel

> Everything below is ready to paste into Hashnode. Images to upload are in
> `docs/figures/`: `speedup.png`, `latency_vs_duration.png`,
> `melspec_example.png` — insert them where marked.

---

Log-mel spectrograms are the bread and butter of audio ML. Keyword spotting,
speech recognition front ends, my own cough-detection work — they all start by
turning a waveform into that familiar orange-and-purple heatmap. So when your
model takes 2 ms to run and your feature extraction takes a chunk of that
budget, the spectrogram is worth optimizing.

I wanted to learn CUDA properly, so I set myself a concrete goal: **beat
torchaudio's GPU mel-spectrogram path by at least 2× on typical short audio
clips.** I ended up at **~3×**, and the way I got there taught me more about
how GPUs actually spend their time than any tutorial had.

**[IMAGE: speedup.png]**

## The setup

The pipeline everyone uses:

1. Split the waveform into overlapping frames (e.g. 512 samples, hop 160)
2. Multiply each frame by a Hann window
3. FFT each frame, keep the magnitude squared (power spectrogram)
4. Multiply by a mel filterbank matrix — 64 triangular filters spaced by
   `mel(f) = 2595·log10(1 + f/700)`
5. Take the log

torchaudio does all of this on GPU already, and cuFFT (which it uses under the
hood) is extremely good. So where's the win?

## Where torchaudio spends its time (hint: not computing)

I profiled `torchaudio.transforms.MelSpectrogram` with the PyTorch profiler
and counted **10 separate CUDA kernel launches per call**: padding, framing,
windowing, FFT stages, magnitude, a dense matmul for the mel projection, log.
Between the big stages, intermediate tensors — the full power spectrogram, the
pre-log mel spectrogram — get written to GPU DRAM and read back.

Then I looked at what the GPU was doing during my fused kernel with Nsight
Compute:

- SM (compute) throughput: **~5%**
- DRAM throughput: **~5%**

Five percent. For a 3.5-second clip at typical front-end settings, the actual
math is *tiny* compared to what the GPU can do. The pipeline isn't
compute-bound or even bandwidth-bound — it's **launch-bound**. Every kernel
launch costs microseconds of overhead, and with 10 launches doing microseconds
of work each, overhead *is* the workload.

That reframes the optimization problem completely. Writing a faster FFT than
cuFFT? Hopeless. Launching fewer kernels and touching DRAM less? Very doable.

## The design

My pipeline is 4 launches:

```
waveform → reflection pad (ATen)
        → frame + window          [custom kernel]
        → batched R2C FFT         [cuFFT, one plan for ALL frames]
        → magnitude² + mel + log  [ONE fused kernel]
```

Two ideas do the heavy lifting.

### Idea 1: keep the power spectrum in shared memory

The fused kernel assigns **one thread block per STFT frame**. In phase one,
the block's 256 threads cooperatively compute `|X|²` for the frame's 257
frequency bins — writing them into **shared memory**, not global memory. The
power spectrogram, which torchaudio materializes fully in DRAM, never exists
in DRAM at all.

```cpp
extern __shared__ float power[];              // 257 floats, on-chip
for (int i = threadIdx.x; i < n_freq; i += blockDim.x) {
    cufftComplex c = frame_fft[i];
    power[i] = c.x * c.x + c.y * c.y;
}
__syncthreads();
```

### Idea 2: mel filters are sparse — skip the matmul

The textbook implementation multiplies the power spectrogram by a dense
`(64 × 257)` filterbank matrix. But look at a mel filterbank: each triangular
filter is nonzero over only a handful of contiguous bins. Most of that matrix
is zeros.

So I store the filterbank CSR-style — a flat `weights` array plus per-filter
`offsets` and `start_bin` — and in phase two, each thread takes a mel filter
and computes a short dot product over just its nonzero range, reading the
power values from shared memory:

```cpp
for (int m = threadIdx.x; m < n_mels; m += blockDim.x) {
    int begin = mel_offsets[m], end = mel_offsets[m + 1];
    int sb = mel_start_bin[m];
    float acc = 0.0f;
    for (int k = 0; k < end - begin; ++k)
        acc += __ldg(&mel_weights[begin + k]) * power[sb + k];
    out[frame * n_mels + m] = logf(acc + eps);
}
```

The log is applied right there — no separate log kernel, no pre-log
intermediate buffer.

## Results

NVIDIA GB10 (DGX Spark), CUDA 13.0, PyTorch 2.12.1. 100 timed iterations with
CUDA events after warmup; config sr=16 kHz, n_fft=512, hop=160, n_mels=64.

| clip | mine (fused) | torchaudio GPU | librosa CPU | speedup |
|---|---|---|---|---|
| 0.5 s | 20.0 µs | 62.1 µs | 336 µs | **3.10×** |
| 1.0 s | 20.4 µs | 61.7 µs | 401 µs | **3.02×** |
| 3.5 s | 21.2 µs | 60.8 µs | 781 µs | **2.87×** |
| 10 s | 27.1 µs | 84.0 µs | 1735 µs | **3.10×** |

**[IMAGE: latency_vs_duration.png]**

Notice the speedup is basically *flat* across clip lengths. That's the
launch-bound story confirmed: until the GPU actually saturates, latency —
not throughput — sets the floor, so cutting launches from 10 to 4 pays the
same dividend everywhere. I also built a "naive" version of my own pipeline
(separate magnitude / cuBLAS matmul / log kernels, 7 launches) as a controlled
baseline; fusion alone accounts for a further 1.4–1.5× on top of it.

And the output is not approximately right — it matches
`torchaudio.transforms.MelSpectrogram` to a max absolute error of **2×10⁻⁵**
across five different STFT configurations.

**[IMAGE: melspec_example.png]**

## What I learned

1. **Profile before optimizing, and profile the right metric.** I assumed I'd
   be writing clever tiled matmuls. Nsight Compute's speed-of-light section
   said the GPU was 95% idle — the enemy was launch count, not FLOPs.
2. **cuFFT's batch interface is the whole game for STFT.** One
   `cufftPlanMany` covering all frames, cached across calls, turns hundreds
   of FFTs into one launch.
3. **Shared memory is for skipping DRAM, not just for tiling.** The power
   spectrum lives for exactly one kernel — so give it a lifetime of one
   kernel. Data that never hits DRAM is free to re-read.
4. **Exploit structure the general-purpose library can't assume.** torchaudio
   has to do a dense matmul because its filterbank is an arbitrary matrix. I
   know mine is triangular filters with compact support, so I get to skip
   ~98% of the multiply.
5. **Kernel launch overhead is real money at the microsecond scale.** 10
   launches → 4 launches was worth more than every arithmetic trick combined.

The whole thing is a pip-installable PyTorch extension —
`MelSpectrogramCUDA(...)` is a drop-in module. Code, benchmarks, and a 2-page
technical note are on GitHub: **[LINK: your GitHub repo URL]**.

---

*Built as a learning project on a DGX Spark. If you're doing audio ML on edge
GPUs and your feature extraction shows up in your latency budget, kernel
fusion is probably the cheapest 3× you'll ever get.*
