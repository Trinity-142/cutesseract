#pragma once

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

#define CUDA_CHECK(call) cudaCheckCall(call, __FILE__, __LINE__)
#define CURAND_CHECK(call) curandCheckCall(call, __FILE__, __LINE__)
