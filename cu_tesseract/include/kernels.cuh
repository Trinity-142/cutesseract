#pragma once

#include <cuda_runtime.h>
#include "dtypes.cuh"
#include "matrix.cuh"
#include "utils.cuh"

/*

RTX 3060 math https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3682
 
SM-count 28

Warp-size 32 threads
Active warps in SM - 4
Total warps active - 112

Max waprs in SM - 48
Max Block per SM - 16!!

Tensor Core count - 112

SMEM per SM - 128kb
equal SMEM per Warp - 32kb

*/

// Tiled Shared Memory Kernel
template <size_t BS>
static __global__ void _gemm_nnn_block_simple(
    size_t N,
    fp32 *A, // row-wise
    fp32 *B, // row-wise
    fp32 *C  // row-wise
) {
    size_t row = blockIdx.y * BS + threadIdx.y;
    size_t col = blockIdx.x * BS + threadIdx.x;

    fp32 sum = 0.0;

    __shared__ fp32 block_a[BS][BS];
    __shared__ fp32 block_b[BS][BS];

    for (size_t s = 0; s < (N / BS); s++) {
        block_a[threadIdx.y][threadIdx.x] = A[row * N + (s * BS + threadIdx.x)];
        block_b[threadIdx.y][threadIdx.x] = B[(s * BS + threadIdx.y) * N + col];

        __syncthreads();

        for (size_t k = 0; k < BS; k++)
            sum += block_a[threadIdx.y][k] * block_b[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple Element-wise Kernel
// n*k x k*m = n*m
static __global__ void _gemm_nkm_simple(
    size_t N, size_t K, size_t M,
    fp32 *A, // row-wise
    fp32 *B, // row-wise
    fp32 *C  // row-wise
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        fp32 sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

// Launchers using runtime dimensions
__host__ void simple_launcher(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    auto [N, K] = A.shape();
    auto [K2, M] = B.shape();
    A.cuda(); B.cuda(); C.cuda();

    dim3 block_dim(16, 16);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);

    _gemm_nkm_simple<<<grid_dim, block_dim>>>(N, K, M, A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void blocked_launcher(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    auto [N, K] = A.shape();
    const size_t BS = 16;
    assert(N == K && "Blocked kernel currently supports square matrices only");
    assert(N % BS == 0 && "Size must be multiple of block size");

    A.cuda(); B.cuda(); C.cuda();

    dim3 block_dim(BS, BS);
    dim3 grid_dim(N / BS, N / BS);

    _gemm_nnn_block_simple<BS><<<grid_dim, block_dim>>>(N, A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}
