#include "kernels.cuh"


template <typename T, size_t N>
__global__ void add_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * ldc + col] = A[row * lda + col] + B[row * ldb + col];
    }
}

template <typename T, size_t N>
__global__ void sub_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * ldc + col] = A[row * lda + col] - B[row * ldb + col];
    }
}

template <typename T, size_t N>
__global__ void gemm_base_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        T sum = 0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = sum;
    }
}
