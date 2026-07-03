"""cuda_mel: fused STFT + mel filterbank CUDA kernel, callable from Python.

    import torch, cuda_mel
    mel = cuda_mel.MelSpectrogramCUDA(sample_rate=16000, n_fft=512,
                                       hop_length=160, n_mels=64).cuda()
    log_mel = mel(waveform)  # waveform: 1D float32 CUDA tensor
"""

import torch
import torch.nn as nn

from .melfb import build_mel_filterbank

try:
    import cuda_mel_ext
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cuda_mel_ext extension not built. Run `pip install .` from the "
        "repo root on a machine with CUDA + nvcc available."
    ) from e


class MelSpectrogramCUDA(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop_length: int | None = None, n_mels: int = 64,
                 f_min: float = 0.0, f_max: float | None = None,
                 eps: float = 1e-6, fused: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.n_mels = n_mels
        self.eps = eps
        self.fused = fused

        window = torch.hann_window(n_fft, periodic=True)
        dense, weights, offsets, start_bin = build_mel_filterbank(
            sample_rate, n_fft, n_mels, f_min, f_max)

        self.register_buffer("window", window)
        self.register_buffer("mel_fb_t", torch.from_numpy(dense))
        self.register_buffer("mel_weights", torch.from_numpy(weights))
        self.register_buffer("mel_offsets", torch.from_numpy(offsets))
        self.register_buffer("mel_start_bin", torch.from_numpy(start_bin))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.is_cuda, "waveform must be a CUDA tensor"
        waveform = waveform.contiguous().float()
        if self.fused:
            return cuda_mel_ext.fused_mel_spectrogram(
                waveform, self.window, self.mel_weights, self.mel_offsets,
                self.mel_start_bin, self.n_mels, self.n_fft, self.hop_length,
                self.eps)
        return cuda_mel_ext.naive_mel_spectrogram(
            waveform, self.window, self.mel_fb_t, self.n_fft, self.hop_length,
            self.eps)
