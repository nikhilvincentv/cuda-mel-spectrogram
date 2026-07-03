#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                                cudaGetErrorString(_err) + " at " + __FILE__ + \
                                ":" + std::to_string(__LINE__));               \
    }                                                                         \
  } while (0)

#define CUFFT_CHECK(expr)                                                     \
  do {                                                                        \
    cufftResult _err = (expr);                                                \
    if (_err != CUFFT_SUCCESS) {                                              \
      throw std::runtime_error(std::string("cuFFT error: ") +                 \
                                std::to_string(_err) + " at " + __FILE__ +     \
                                ":" + std::to_string(__LINE__));               \
    }                                                                         \
  } while (0)
