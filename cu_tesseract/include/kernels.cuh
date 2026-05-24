#pragma once

#include <cuda_runtime.h>
#include <mma.h>

#include "dtypes.cuh"
#include "matrix.cuh"
#include "utils.cuh"
using namespace nvcuda;


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

template <typename T>
static __global__ void _gemm_nnn_block_simple(T *A, // row-wise
                                              T *B, // row-wise
                                              T *C, // row-wise
                                              size_t N, size_t BS) {
  size_t row = blockIdx.y * BS + threadIdx.y;
  size_t col = blockIdx.x * BS + threadIdx.x;

  T sum = 0.0;

  extern __shared__ char smem[];
  T *block_a = (T *)smem;
  T *block_b = (T *)(smem + BS * BS * sizeof(T));

  for (size_t s = 0; s < (N / BS); s++) {
    block_a[threadIdx.y * BS + threadIdx.x] =
        A[row * N + (s * BS + threadIdx.x)];
    block_b[threadIdx.y * BS + threadIdx.x] =
        B[(s * BS + threadIdx.y) * N + col];

    __syncthreads();

    for (size_t k = 0; k < BS; k++)
      sum += block_a[threadIdx.y * BS + k] * block_b[k * BS + threadIdx.x];

    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = sum;
  }
}


template <typename T>
__host__ void _gemm_nn_block_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C,
                                      size_t BS = 16) {
  size_t N = A.shape().first;
  assert(A.shape().first == N && A.shape().second == N);
  assert(B.shape().first == N && B.shape().second == N);
  assert(C.shape().first == N && C.shape().second == N);

  assert((N % BS) == 0);

  assert(A.get_layout() == DataLayout::ROW_WISE);
  assert(B.get_layout() == DataLayout::ROW_WISE);
  assert(C.get_layout() == DataLayout::ROW_WISE);

  A.cuda();
  B.cuda();
  C.cuda();

  dim3 block_dim(BS, BS); // x, y
  dim3 grid_dim(N / BS, N / BS);

  size_t shared_mem_size = 2 * BS * BS * sizeof(T);
  cudaFuncSetCacheConfig(_gemm_nnn_block_simple<T>, cudaFuncCachePreferShared);
  _gemm_nnn_block_simple<T><<<grid_dim, block_dim, shared_mem_size>>>(
      A.item(), B.item(), C.item(), N, BS);
  CUDA_CHECK(cudaDeviceSynchronize());
}


template <typename T>
static __global__ void _gemm_nkm_simple(T *A, // row-wise
                                        T *B, // row-wise
                                        T *C, // row-wise
                                        size_t N, size_t K, size_t M) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    T sum = 0;
    for (size_t i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * M + col];
    }
    C[row * M + col] = sum;
  }
}


template <typename T> // n*k x k*m = n*m
__host__ void _gemm_nkm_simple_launcher(Matrix<T> &A, Matrix<T> &B,
                                        Matrix<T> &C) {
  size_t N = A.shape().first;
  size_t K = A.shape().second;
  size_t M = B.shape().second;

  assert(B.shape().first == K);
  assert(C.shape().first == N && C.shape().second == M);

  assert(A.get_layout() == DataLayout::ROW_WISE);
  assert(B.get_layout() == DataLayout::ROW_WISE);
  assert(C.get_layout() == DataLayout::ROW_WISE);

  A.cuda();
  B.cuda();
  C.cuda();

  dim3 block_dim(16, 16);
  dim3 grid_dim((M + block_dim.x - 1) / block_dim.x,
                (N + block_dim.y - 1) / block_dim.y);

  cudaFuncSetCacheConfig(_gemm_nkm_simple<T>, cudaFuncCachePreferL1);
  _gemm_nkm_simple<T>
      <<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
  CUDA_CHECK(cudaDeviceSynchronize());
}


// https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
constexpr int tileSize = 16;
constexpr int threadsPerWarp = 32;
constexpr size_t warpBlockSize = 2;
constexpr size_t PAD = 8;
// A(n*k) x B(k*m) = c(n*m)
template <bool IS_ALIGNED>
static __global__ void _gemm_nkm_wmma_simple(
    fp16 *A,
    fp16 *B,
    fp32 *C,
    size_t N, size_t K, size_t M
) {
    size_t threadId = threadIdx.y * blockDim.x + threadIdx.x;
    size_t warpId = threadId / threadsPerWarp;                          // [0..3] 2x2 warps for each thread block
    size_t warpRow = warpId / warpBlockSize;                            // [0, 1]
    size_t warpCol = warpId % warpBlockSize;                            // [0, 1]
    // top left block corner
    size_t globalBlockRow = blockIdx.y * (warpBlockSize * tileSize);
    size_t globalBlockCol = blockIdx.x * (warpBlockSize * tileSize);

    // 32x16 from A * 16x32 from B = 32x32 in C calcs each thread block. 4 mmuls of 16x16 tiles -> 1 mmul 16x16 for each warp
    __shared__ fp16 block_A[warpBlockSize * tileSize][tileSize + PAD];  // PAD to resolve smem bank conflicts (https://modal.com/gpu-glossary/perf/bank-conflict)
    __shared__ fp16 block_B[tileSize][warpBlockSize * tileSize + PAD];

    wmma::fragment<wmma::matrix_a, tileSize, tileSize, tileSize, fp16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, tileSize, tileSize, tileSize, fp16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, tileSize, tileSize, tileSize, fp32> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (size_t i = 0; i < K; i += tileSize) {
        // 32x16 = 512 fp16 elements, 128 threads in block -> each thread should store 4 fp16 elements (or one fp64) to smem
        size_t row_A = threadId / 4;                                    // 16 / 4 = 4 threads per row
        size_t col_A = (threadId % 4) * 4;
        size_t global_row_A = globalBlockRow + row_A;
        size_t global_col_A = i + col_A;
        *(reinterpret_cast<fp64*>(&block_A[row_A][col_A])) = safe_load<IS_ALIGNED>(A, global_row_A, global_col_A, N, K, K);

        size_t row_B = threadId / 8;                                    // 32 / 4 = 8 threads per row
        size_t col_B = (threadId % 8) * 4;
        size_t global_row_B = i + row_B;
        size_t global_col_B = globalBlockCol + col_B;
       *(reinterpret_cast<fp64*>(&block_B[row_B][col_B])) = safe_load<IS_ALIGNED>(B, global_row_B, global_col_B, K, M, N);
        __syncthreads();

        fp16* warp_A = &block_A[warpRow * tileSize][0];                 // top left corner of 16x16 tile from A for that warp
        fp16* warp_B = &block_B[0][warpCol * tileSize];                 // top left corner of 16x16 tile from B for that warp
        wmma::load_matrix_sync(a_frag, warp_A, tileSize + PAD);
        wmma::load_matrix_sync(b_frag, warp_B, warpBlockSize * tileSize + PAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    size_t globalRow = globalBlockRow + warpRow * tileSize;
    size_t globalCol = globalBlockCol + warpCol * tileSize;
    if constexpr (IS_ALIGNED) {
        wmma::store_matrix_sync(C + globalRow * M + globalCol, c_frag, M, wmma::mem_row_major);
    } else {
        __shared__ float block_C[warpBlockSize * tileSize][warpBlockSize * tileSize];
        float* warp_C = &block_C[warpRow * tileSize][warpCol * tileSize];
        wmma::store_matrix_sync(warp_C, c_frag, warpBlockSize * tileSize, wmma::mem_row_major);
        __syncthreads();

        for (size_t i = threadId; i < (warpBlockSize * tileSize) * (warpBlockSize * tileSize); i += blockDim.x * blockDim.y) {
            size_t local_r = i / (warpBlockSize * tileSize);
            size_t local_c = i % (warpBlockSize * tileSize);
            size_t g_r = globalBlockRow + local_r;
            size_t g_c = globalBlockCol + local_c;
            if (g_r < N && g_c < M) {
                C[g_r * M + g_c] = block_C[local_r][local_c];
            }
        }
    }
}


__host__ void _gemm_nkm_wmma_launcher(Matrix<fp16> &A, Matrix<fp16> &B, Matrix<fp32> &C) {
    size_t N = A.shape().first;
    size_t K = A.shape().second;
    size_t M = B.shape().second;

    assert(A.get_layout() == DataLayout::ROW_WISE);
    assert(B.get_layout() == DataLayout::ROW_WISE);
    assert(C.get_layout() == DataLayout::ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    bool is_aligned = (K % tileSize == 0) && (N % (tileSize * warpBlockSize) == 0) && (M % (tileSize * warpBlockSize) == 0);
    dim3 block_dim(warpBlockSize * threadsPerWarp, warpBlockSize);                  // x:[0..63], y:[0,1] -> 2x2 warp grid
    dim3 grid_dim((M + (tileSize * warpBlockSize - 1)) / (tileSize * warpBlockSize),
                (N + (tileSize * warpBlockSize - 1)) / (tileSize * warpBlockSize));
    static_assert(warpBlockSize * warpBlockSize * threadsPerWarp <= 1024);
    if (is_aligned) {
        _gemm_nkm_wmma_simple<true><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
    } else {
        _gemm_nkm_wmma_simple<false><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
