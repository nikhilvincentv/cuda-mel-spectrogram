// pybind11 / torch extension entry points.
//
// Exposes two functions to Python:
//   naive_mel_spectrogram  - frame+window kernel -> cuFFT -> magnitude kernel
//                             -> cuBLAS sgemm mel multiply -> log kernel
//                             (four kernel launches + two intermediate
//                             global-memory buffers: power spectrogram and
//                             pre-log mel spectrogram)
//   fused_mel_spectrogram  - frame+window kernel -> cuFFT -> single fused
//                             kernel (magnitude + sparse mel dot product +
//                             log, power spectrum lives only in shared
//                             memory)
//
// Both take a raw waveform and a precomputed mel filterbank (constructed on
// the Python side in python/cuda_mel/melfb.py to match torchaudio's HTK mel
// scale exactly) and return a log-mel spectrogram tensor of shape
// (num_frames, n_mels).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <unordered_map>
#include "common.h"

void launch_frame_and_window(const float* padded_waveform, const float* window,
                              float* framed_out, int n_fft, int hop_length,
                              int num_frames, cudaStream_t stream);
void launch_magnitude(const cufftComplex* fft_out, float* power, int total_bins,
                       cudaStream_t stream);
void launch_log(float* x, int n, float eps, cudaStream_t stream);
void launch_fused_mel_sparse(const cufftComplex* fft_out, const float* mel_weights,
                              const int* mel_offsets, const int* mel_start_bin,
                              float* out, int n_freq, int n_mels, int num_frames,
                              float eps, cudaStream_t stream);

namespace {

// cuFFT plans are expensive to create (device allocations + kernel
// selection). Real workloads call this repeatedly with the same clip length,
// so we cache plans by (n_fft, num_frames) exactly like PyTorch/torchaudio's
// own internal cuFFT plan cache.
cufftHandle get_or_create_plan(int n_fft, int num_frames) {
  static std::unordered_map<int64_t, cufftHandle> cache;
  int64_t key = (int64_t)n_fft * 1000000007LL + num_frames;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  cufftHandle plan;
  int n[1] = {n_fft};
  CUFFT_CHECK(cufftPlanMany(&plan, 1, n, nullptr, 1, n_fft, nullptr, 1,
                             n_fft / 2 + 1, CUFFT_R2C, num_frames));
  cache[key] = plan;
  return plan;
}

int compute_num_frames(int padded_len, int n_fft, int hop_length) {
  return 1 + (padded_len - n_fft) / hop_length;
}

}  // namespace

torch::Tensor naive_mel_spectrogram(torch::Tensor waveform, torch::Tensor window,
                                     torch::Tensor mel_fb_t, int64_t n_fft,
                                     int64_t hop_length, double eps) {
  TORCH_CHECK(waveform.is_cuda() && waveform.dtype() == torch::kFloat32);
  TORCH_CHECK(window.is_cuda() && window.dtype() == torch::kFloat32);
  TORCH_CHECK(mel_fb_t.is_cuda() && mel_fb_t.dtype() == torch::kFloat32);

  auto stream = at::cuda::getCurrentCUDAStream();
  int pad = n_fft / 2;
  auto padded = at::reflection_pad1d(waveform.view({1, 1, -1}), {pad, pad}).view(-1);
  int padded_len = padded.size(0);
  int num_frames = compute_num_frames(padded_len, n_fft, hop_length);
  int n_freq = n_fft / 2 + 1;
  int64_t n_mels = mel_fb_t.size(1);

  auto opts_f = waveform.options();
  auto framed = torch::empty({num_frames, n_fft}, opts_f);
  auto fft_out = torch::empty({num_frames, n_freq},
                               opts_f.dtype(torch::kComplexFloat));
  auto power = torch::empty({num_frames, n_freq}, opts_f);
  auto out = torch::empty({num_frames, n_mels}, opts_f);

  launch_frame_and_window(padded.data_ptr<float>(), window.data_ptr<float>(),
                           framed.data_ptr<float>(), n_fft, hop_length, num_frames,
                           stream.stream());

  cufftHandle plan = get_or_create_plan(n_fft, num_frames);
  CUFFT_CHECK(cufftSetStream(plan, stream.stream()));
  CUFFT_CHECK(cufftExecR2C(
      plan, framed.data_ptr<float>(),
      reinterpret_cast<cufftComplex*>(fft_out.data_ptr<c10::complex<float>>())));

  launch_magnitude(
      reinterpret_cast<cufftComplex*>(fft_out.data_ptr<c10::complex<float>>()),
      power.data_ptr<float>(), num_frames * n_freq, stream.stream());

  cublasHandle_t blas = at::cuda::getCurrentCUDABlasHandle();
  float alpha = 1.0f, beta = 0.0f;
  // Row-major C(F,M) = A(F,K) * B(K,M), computed via the standard cuBLAS
  // row-major trick: call column-major sgemm(B, A) with swapped dims.
  cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n_mels, num_frames, n_freq, &alpha,
              mel_fb_t.data_ptr<float>(), n_mels, power.data_ptr<float>(), n_freq,
              &beta, out.data_ptr<float>(), n_mels);

  launch_log(out.data_ptr<float>(), num_frames * n_mels, (float)eps, stream.stream());
  return out;
}

torch::Tensor fused_mel_spectrogram(torch::Tensor waveform, torch::Tensor window,
                                     torch::Tensor mel_weights,
                                     torch::Tensor mel_offsets,
                                     torch::Tensor mel_start_bin, int64_t n_mels,
                                     int64_t n_fft, int64_t hop_length, double eps) {
  TORCH_CHECK(waveform.is_cuda() && waveform.dtype() == torch::kFloat32);
  TORCH_CHECK(window.is_cuda() && window.dtype() == torch::kFloat32);
  TORCH_CHECK(mel_weights.is_cuda() && mel_weights.dtype() == torch::kFloat32);
  TORCH_CHECK(mel_offsets.is_cuda() && mel_offsets.dtype() == torch::kInt32);
  TORCH_CHECK(mel_start_bin.is_cuda() && mel_start_bin.dtype() == torch::kInt32);

  auto stream = at::cuda::getCurrentCUDAStream();
  int pad = n_fft / 2;
  auto padded = at::reflection_pad1d(waveform.view({1, 1, -1}), {pad, pad}).view(-1);
  int padded_len = padded.size(0);
  int num_frames = compute_num_frames(padded_len, n_fft, hop_length);
  int n_freq = n_fft / 2 + 1;

  auto opts_f = waveform.options();
  auto framed = torch::empty({num_frames, n_fft}, opts_f);
  auto fft_out = torch::empty({num_frames, n_freq},
                               opts_f.dtype(torch::kComplexFloat));
  auto out = torch::empty({num_frames, n_mels}, opts_f);

  launch_frame_and_window(padded.data_ptr<float>(), window.data_ptr<float>(),
                           framed.data_ptr<float>(), n_fft, hop_length, num_frames,
                           stream.stream());

  cufftHandle plan = get_or_create_plan(n_fft, num_frames);
  CUFFT_CHECK(cufftSetStream(plan, stream.stream()));
  CUFFT_CHECK(cufftExecR2C(
      plan, framed.data_ptr<float>(),
      reinterpret_cast<cufftComplex*>(fft_out.data_ptr<c10::complex<float>>())));

  launch_fused_mel_sparse(
      reinterpret_cast<cufftComplex*>(fft_out.data_ptr<c10::complex<float>>()),
      mel_weights.data_ptr<float>(), mel_offsets.data_ptr<int>(),
      mel_start_bin.data_ptr<int>(), out.data_ptr<float>(), n_freq, n_mels,
      num_frames, (float)eps, stream.stream());

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_mel_spectrogram", &naive_mel_spectrogram,
        "Naive multi-kernel STFT + mel spectrogram (CUDA)");
  m.def("fused_mel_spectrogram", &fused_mel_spectrogram,
        "Fused tiled STFT + mel spectrogram (CUDA)");
}
