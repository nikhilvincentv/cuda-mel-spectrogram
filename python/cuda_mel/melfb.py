"""Mel filterbank construction, matching torchaudio.functional.melscale_fbanks
with mel_scale="htk", norm=None (torchaudio's MelSpectrogram default).

Produces two representations of the same filterbank:
  - dense (n_freq, n_mels) matrix for the naive cuBLAS-based kernel
  - compact sparse (weights/offsets/start_bin) arrays for the fused kernel,
    exploiting the fact that each triangular mel filter has compact support
    over a small contiguous range of frequency bins.
"""

import numpy as np


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                          f_min: float = 0.0, f_max: float | None = None):
    """Returns dense (n_freq, n_mels) float32 filterbank plus compact arrays.

    dense[k, m] is the weight applied to FFT bin k for mel filter m -- this is
    mel_fb_t as expected by naive_mel_spectrogram (already transposed to
    (n_freq, n_mels) so the CUDA binding can do a plain row-major matmul).
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freq = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freq)

    m_min, m_max = hz_to_mel(np.array([f_min, f_max]))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)  # n_mels + 2 boundary frequencies

    dense = np.zeros((n_freq, n_mels), dtype=np.float32)
    weights_list, offsets, start_bins = [], [0], []

    for m in range(n_mels):
        f_left, f_center, f_right = f_pts[m], f_pts[m + 1], f_pts[m + 2]

        left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-10)
        right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-10)
        tri = np.maximum(0.0, np.minimum(left_slope, right_slope)).astype(np.float32)

        nonzero = np.nonzero(tri)[0]
        if len(nonzero) == 0:
            start_bins.append(0)
            offsets.append(offsets[-1])
            continue
        lo, hi = nonzero[0], nonzero[-1] + 1  # compact contiguous support
        dense[lo:hi, m] = tri[lo:hi]
        weights_list.append(tri[lo:hi])
        start_bins.append(int(lo))
        offsets.append(offsets[-1] + (hi - lo))

    weights = np.concatenate(weights_list).astype(np.float32) if weights_list \
        else np.zeros((0,), dtype=np.float32)
    offsets = np.array(offsets, dtype=np.int32)
    start_bins = np.array(start_bins, dtype=np.int32)

    return dense, weights, offsets, start_bins
