"""Correctness tests: compare cuda_mel's naive and fused kernels against
torchaudio.transforms.MelSpectrogram on GPU.

Run on a CUDA machine:  python tests/test_correctness.py
"""

import sys

import torch
import torchaudio

import cuda_mel

CONFIGS = [
    # (sample_rate, n_fft, hop, n_mels, duration_s)
    (16000, 512, 160, 64, 1.0),
    (16000, 512, 160, 64, 3.5),
    (16000, 1024, 256, 80, 2.0),
    (22050, 2048, 512, 128, 5.0),
    (16000, 400, 160, 40, 0.5),
]


def reference_logmel(waveform, sr, n_fft, hop, n_mels, eps):
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        power=2.0, center=True, pad_mode="reflect", norm=None,
        mel_scale="htk", window_fn=torch.hann_window,
    ).cuda()
    # (n_mels, num_frames) -> (num_frames, n_mels) to match our layout
    return torch.log(ref(waveform) + eps).T.contiguous()


def run():
    eps = 1e-6
    torch.manual_seed(0)
    n_fail = 0

    for sr, n_fft, hop, n_mels, dur in CONFIGS:
        waveform = torch.randn(int(sr * dur), device="cuda")

        ours_fused = cuda_mel.MelSpectrogramCUDA(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            eps=eps, fused=True).cuda()(waveform)
        ours_naive = cuda_mel.MelSpectrogramCUDA(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            eps=eps, fused=False).cuda()(waveform)
        ref = reference_logmel(waveform, sr, n_fft, hop, n_mels, eps)

        for name, ours in [("fused", ours_fused), ("naive", ours_naive)]:
            same_shape = ours.shape == ref.shape
            max_abs = (ours - ref).abs().max().item() if same_shape else float("inf")
            ok = same_shape and torch.allclose(ours, ref, rtol=1e-3, atol=1e-4)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {name:5s} sr={sr} n_fft={n_fft} hop={hop} "
                  f"n_mels={n_mels} dur={dur}s  shape={tuple(ours.shape)} "
                  f"max_abs_err={max_abs:.2e}")
            n_fail += 0 if ok else 1

    if n_fail:
        print(f"\n{n_fail} check(s) FAILED")
        sys.exit(1)
    print("\nAll correctness checks passed.")


if __name__ == "__main__":
    run()
