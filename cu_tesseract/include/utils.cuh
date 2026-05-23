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

__global__ inline void castFp32ToFp16(const fp32* in, half* out, const size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

template <bool IS_ALIGNED>
__device__ __forceinline__ fp64 safe_load(
    half* mat,
    size_t global_row,
    size_t global_col,
    size_t max_rows,
    size_t max_cols,
    size_t stride
) {
    fp64 val = 0;
    if constexpr (IS_ALIGNED) {
        val = *(reinterpret_cast<const fp64*>(&mat[global_row * stride + global_col]));
    } else if (global_row < max_rows) {
        half* val_half = reinterpret_cast<half*>(&val);
        #pragma unroll
        for (size_t j = 0; j < 4; ++j) {
            if (global_col + j < max_cols) {
                val_half[j] = mat[global_row * stride + global_col + j];
            }
        }
    }
    return val;
}

#define CUDA_CHECK(call) cudaCheckCall(call, __FILE__, __LINE__)
#define CURAND_CHECK(call) curandCheckCall(call, __FILE__, __LINE__)
