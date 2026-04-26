#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <string.h>
#include <string>

inline void cudaCheckCall(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::string err_msg = std::string(cudaGetErrorString(code)) + " in " +
                          file + ":" + std::to_string(line);
    throw std::runtime_error(err_msg);
  }
}

inline void curandCheckCall(curandStatus_t code, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {
    std::string err_msg = "CURAND Error " + std::to_string(code) + " in " +
                          file + ":" + std::to_string(line);
    throw std::runtime_error(err_msg);
  }
}

__global__ inline void cast_fp32_to_fp16(const fp32* in, half* out, const size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

#define CUDA_CHECK(call) cudaCheckCall(call, __FILE__, __LINE__)
#define CURAND_CHECK(call) curandCheckCall(call, __FILE__, __LINE__)
