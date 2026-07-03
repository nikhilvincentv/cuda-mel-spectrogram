// CUDA kernels for STFT framing/windowing, power-spectrum magnitude,
// and the fused mel-filterbank + log kernel.
//
// Pipeline (naive):
//   frame_and_window_kernel -> cufftExecR2C (batched) -> magnitude_kernel
//   -> cuBLAS sgemm (mel multiply) -> log_kernel
//
// Pipeline (fused):
//   frame_and_window_kernel -> cufftExecR2C (batched) -> fused_mel_kernel_sparse
//   (magnitude + sparse mel dot-product + log, all in one launch, power
//   spectrum kept in shared memory and never touches global memory)

#include "common.h"

// One block per frame. Each thread strides over n_fft samples.
// Coalesced: adjacent threads read adjacent waveform samples / write
// adjacent output samples.
__global__ void frame_and_window_kernel(const float* __restrict__ padded_waveform,
                                         const float* __restrict__ window,
                                         float* __restrict__ framed_out, int n_fft,
                                         int hop_length, int num_frames) {
  int frame = blockIdx.x;
  if (frame >= num_frames) return;
  const float* src = padded_waveform + (size_t)frame * hop_length;
  float* dst = framed_out + (size_t)frame * n_fft;
  for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
    dst[i] = src[i] * window[i];
  }
}

// Naive path: materialize the full power spectrogram in global memory so it
// can be fed to cuBLAS sgemm for the mel multiply.
__global__ void magnitude_kernel(const cufftComplex* __restrict__ fft_out,
                                  float* __restrict__ power, int total_bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_bins) return;
  cufftComplex c = __ldg(&fft_out[idx]);
  power[idx] = c.x * c.x + c.y * c.y;
}

// Naive path: elementwise log(x + eps) applied after the cuBLAS mel multiply.
__global__ void log_kernel(float* __restrict__ x, int n, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  x[idx] = logf(x[idx] + eps);
}

// Fused path. One block per frame.
//   Phase 1: compute |FFT|^2 for this frame directly into shared memory
//            (never written to global memory).
//   Phase 2: each thread owns one or more mel filters. Mel filterbanks have
//            compact triangular support, so we only touch the nonzero
//            frequency range for each filter (mel_weights/mel_offsets/
//            mel_start_bin), instead of doing a dense n_mels x n_freq matmul.
extern "C" __global__ void fused_mel_kernel_sparse(
    const cufftComplex* __restrict__ fft_out, const float* __restrict__ mel_weights,
    const int* __restrict__ mel_offsets, const int* __restrict__ mel_start_bin,
    float* __restrict__ out, int n_freq, int n_mels, float eps) {
  extern __shared__ float power[];  // size n_freq

  int frame = blockIdx.x;
  const cufftComplex* frame_fft = fft_out + (size_t)frame * n_freq;

  for (int i = threadIdx.x; i < n_freq; i += blockDim.x) {
    cufftComplex c = frame_fft[i];
    power[i] = c.x * c.x + c.y * c.y;
  }
  __syncthreads();

  for (int m = threadIdx.x; m < n_mels; m += blockDim.x) {
    int begin = mel_offsets[m];
    int end = mel_offsets[m + 1];
    int sb = mel_start_bin[m];
    float acc = 0.0f;
    int len = end - begin;
    for (int k = 0; k < len; ++k) {
      acc += __ldg(&mel_weights[begin + k]) * power[sb + k];
    }
    out[(size_t)frame * n_mels + m] = logf(acc + eps);
  }
}

void launch_frame_and_window(const float* padded_waveform, const float* window,
                              float* framed_out, int n_fft, int hop_length,
                              int num_frames, cudaStream_t stream) {
  int threads = 256;
  frame_and_window_kernel<<<num_frames, threads, 0, stream>>>(
      padded_waveform, window, framed_out, n_fft, hop_length, num_frames);
}

void launch_magnitude(const cufftComplex* fft_out, float* power, int total_bins,
                       cudaStream_t stream) {
  int threads = 256;
  int blocks = (total_bins + threads - 1) / threads;
  magnitude_kernel<<<blocks, threads, 0, stream>>>(fft_out, power, total_bins);
}

void launch_log(float* x, int n, float eps, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  log_kernel<<<blocks, threads, 0, stream>>>(x, n, eps);
}

void launch_fused_mel_sparse(const cufftComplex* fft_out, const float* mel_weights,
                              const int* mel_offsets, const int* mel_start_bin,
                              float* out, int n_freq, int n_mels, int num_frames,
                              float eps, cudaStream_t stream) {
  int threads = 256;
  size_t shmem = (size_t)n_freq * sizeof(float);
  fused_mel_kernel_sparse<<<num_frames, threads, shmem, stream>>>(
      fft_out, mel_weights, mel_offsets, mel_start_bin, out, n_freq, n_mels, eps);
}
