"""Generate figures from bench/results.json:
  - latency_vs_duration.png : grouped bar chart of mean latency per impl
  - speedup.png             : fused-kernel speedup over torchaudio GPU / naive
  - melspec_example.png     : example log-mel spectrogram of a chirp (sanity/visual)

Usage:  python bench/make_plots.py --results bench/results.json --outdir docs/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IMPL_LABELS = {
    "cuda_mel_fused": "cuda_mel (fused)",
    "cuda_mel_naive": "cuda_mel (naive)",
    "torchaudio_gpu": "torchaudio (GPU)",
    "librosa_cpu": "librosa (CPU)",
}
IMPL_COLORS = {
    "cuda_mel_fused": "#76b900",
    "cuda_mel_naive": "#a3d55f",
    "torchaudio_gpu": "#ee4c2c",
    "librosa_cpu": "#888888",
}


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data["meta"], data["results"]


def by_impl(results):
    out = {}
    for r in results:
        out.setdefault(r["impl"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["duration_s"])
    return out


def plot_latency(meta, grouped, outdir):
    durations = [r["duration_s"] for r in next(iter(grouped.values()))]
    x = np.arange(len(durations))
    impls = [i for i in IMPL_LABELS if i in grouped]
    width = 0.8 / len(impls)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, impl in enumerate(impls):
        means = [r["mean_ms"] for r in grouped[impl]]
        stds = [r["std_ms"] for r in grouped[impl]]
        ax.bar(x + j * width, means, width, yerr=stds, capsize=3,
               label=IMPL_LABELS[impl], color=IMPL_COLORS[impl])
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(impls) - 1) / 2)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_xlabel("Audio clip duration")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title(f"Log-mel spectrogram latency — {meta['gpu']}\n"
                 f"sr={meta['sr']}, n_fft={meta['n_fft']}, hop={meta['hop']}, "
                 f"n_mels={meta['n_mels']}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latency_vs_duration.png"), dpi=150)
    print("wrote latency_vs_duration.png")


def plot_speedup(meta, grouped, outdir):
    durations = [r["duration_s"] for r in grouped["cuda_mel_fused"]]
    fused = np.array([r["mean_ms"] for r in grouped["cuda_mel_fused"]])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for baseline, style in [("torchaudio_gpu", "o-"), ("cuda_mel_naive", "s--")]:
        if baseline not in grouped:
            continue
        base = np.array([r["mean_ms"] for r in grouped[baseline]])
        ax.plot(durations, base / fused, style,
                label=f"vs {IMPL_LABELS[baseline]}",
                color=IMPL_COLORS[baseline], linewidth=2)
    ax.axhline(1.0, color="k", linewidth=0.8, alpha=0.5)
    ax.axhline(2.0, color="#76b900", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.text(durations[-1], 2.02, "2x target", ha="right", va="bottom",
            fontsize=8, color="#76b900")
    ax.set_xlabel("Audio clip duration (s)")
    ax.set_ylabel("Speedup of fused kernel (x)")
    ax.set_title(f"Fused-kernel speedup — {meta['gpu']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "speedup.png"), dpi=150)
    print("wrote speedup.png")


def plot_example(meta, outdir):
    import torch
    import cuda_mel
    sr, n_fft, hop, n_mels = meta["sr"], meta["n_fft"], meta["hop"], meta["n_mels"]
    t = torch.linspace(0, 3.5, int(sr * 3.5))
    # chirp 100 Hz -> 4 kHz plus a burst of noise in the middle (cough-ish)
    f0, f1 = 100, 4000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * 3.5))
    wav = torch.sin(phase)
    burst = slice(int(sr * 1.6), int(sr * 1.9))
    wav[burst] += 0.8 * torch.randn(burst.stop - burst.start)
    mel = cuda_mel.MelSpectrogramCUDA(sr, n_fft, hop, n_mels).cuda()
    out = mel(wav.cuda()).cpu().numpy().T  # (n_mels, frames)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(out, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, 3.5, 0, n_mels])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Log-mel spectrogram from the fused CUDA kernel\n"
                 "(100 Hz→4 kHz chirp + noise burst)")
    fig.colorbar(im, ax=ax, label="log power")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "melspec_example.png"), dpi=150)
    print("wrote melspec_example.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.json")
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--skip-example", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    meta, results = load(args.results)
    grouped = by_impl(results)
    plot_latency(meta, grouped, args.outdir)
    plot_speedup(meta, grouped, args.outdir)
    if not args.skip_example:
        plot_example(meta, args.outdir)
