"""Systematic benchmark: cuda_mel (fused + naive) vs torchaudio GPU vs librosa CPU.

For each audio duration, runs `--iters` timed iterations after warmup and
reports mean/std wall-clock latency per call. GPU timing uses CUDA events with
a final synchronize. Results are written to results.json and results.csv.

Usage (on the GPU machine):
    python bench/benchmark.py --iters 100 --out bench/results
"""

import argparse
import csv
import json
import platform
import statistics
import time

import numpy as np
import torch
import torchaudio

import cuda_mel

DURATIONS = [0.5, 1.0, 3.5, 10.0]
SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
EPS = 1e-6


def time_gpu(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def time_cpu(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--librosa-iters", type=int, default=20)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--skip-librosa", action="store_true")
    args = ap.parse_args()

    try:
        import librosa
        have_librosa = not args.skip_librosa
    except ImportError:
        have_librosa = False

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)
    rows = []

    fused = cuda_mel.MelSpectrogramCUDA(SR, N_FFT, HOP, N_MELS, eps=EPS,
                                         fused=True).cuda()
    naive = cuda_mel.MelSpectrogramCUDA(SR, N_FFT, HOP, N_MELS, eps=EPS,
                                         fused=False).cuda()
    ta = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        power=2.0, center=True, pad_mode="reflect", norm=None,
        mel_scale="htk", window_fn=torch.hann_window).cuda()

    for dur in DURATIONS:
        n = int(SR * dur)
        wav_gpu = torch.randn(n, device="cuda")
        wav_np = wav_gpu.cpu().numpy()

        candidates = {
            "cuda_mel_fused": lambda: fused(wav_gpu),
            "cuda_mel_naive": lambda: naive(wav_gpu),
            "torchaudio_gpu": lambda: torch.log(ta(wav_gpu) + EPS),
        }
        for name, fn in candidates.items():
            times = time_gpu(fn, args.iters)
            rows.append(dict(impl=name, duration_s=dur,
                             mean_ms=statistics.mean(times),
                             std_ms=statistics.stdev(times),
                             iters=args.iters))
            print(f"{name:16s} dur={dur:5.1f}s  "
                  f"{rows[-1]['mean_ms']:8.4f} ± {rows[-1]['std_ms']:.4f} ms")

        if have_librosa:
            def lib_fn():
                m = librosa.feature.melspectrogram(
                    y=wav_np, sr=SR, n_fft=N_FFT, hop_length=HOP,
                    n_mels=N_MELS, power=2.0, center=True, htk=True, norm=None)
                return np.log(m + EPS)
            times = time_cpu(lib_fn, args.librosa_iters)
            rows.append(dict(impl="librosa_cpu", duration_s=dur,
                             mean_ms=statistics.mean(times),
                             std_ms=statistics.stdev(times),
                             iters=args.librosa_iters))
            print(f"{'librosa_cpu':16s} dur={dur:5.1f}s  "
                  f"{rows[-1]['mean_ms']:8.4f} ± {rows[-1]['std_ms']:.4f} ms")

    meta = dict(gpu=device_name, torch=torch.__version__,
                torchaudio=torchaudio.__version__,
                platform=platform.platform(),
                sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
    with open(args.out + ".json", "w") as f:
        json.dump(dict(meta=meta, results=rows), f, indent=2)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
