#pragma once

#include <cuda_runtime.h>
#include <crt/mma.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

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
                                              size_t N, size_t BS);

template <typename T>
__host__ void _gemm_nn_block_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C,
                                      size_t BS = 16) {
  size_t N = A.shape().first;
  assert(A.shape().first == N && A.shape().second == N);
  assert(B.shape().first == N && B.shape().second == N);
  assert(C.shape().first == N && C.shape().second == N);

  assert((N % BS) == 0);

  assert(A.get_layout() == ROW_WISE);
  assert(B.get_layout() == ROW_WISE);
  assert(C.get_layout() == ROW_WISE);

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

template <typename T>                         // n*k x k*m = n*m
static __global__ void _gemm_nkm_simple(T *A, // row-wise
                                        T *B, // row-wise
                                        T *C, // row-wise
                                        size_t N, size_t K, size_t M);

template <typename T> // n*k x k*m = n*m
__host__ void _gemm_nkm_simple_launcher(Matrix<T> &A, Matrix<T> &B,
                                        Matrix<T> &C) {
  size_t N = A.shape().first;
  size_t K = A.shape().second;
  size_t M = B.shape().second;

  assert(B.shape().first == K);
  assert(C.shape().first == N && C.shape().second == M);

  assert(A.get_layout() == ROW_WISE);
  assert(B.get_layout() == ROW_WISE);
  assert(C.get_layout() == ROW_WISE);

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

constexpr int tileSize = 16;
constexpr int threadsPerWarp = 32;

// warps topology in block
constexpr size_t WARP_BLOCK_ROWS = 4; // Y
constexpr size_t WARP_BLOCK_COLS = 2; // X
constexpr size_t NUM_WARPS = WARP_BLOCK_ROWS * WARP_BLOCK_COLS;
constexpr size_t NUM_THREADS = NUM_WARPS * threadsPerWarp;

// 32x64 tile in C for each warp
constexpr size_t WARP_TILE_ROWS = 2;  // 2x16 = 32 elements
constexpr size_t WARP_TILE_COLS = 4;  // 4x16 = 64 elements

// 128x128 tile in C for each block
constexpr size_t BLOCK_TILE_ROWS = WARP_BLOCK_ROWS * WARP_TILE_ROWS * tileSize; // 4 * 2 * 16 = 128
constexpr size_t BLOCK_TILE_COLS = WARP_BLOCK_COLS * WARP_TILE_COLS * tileSize; // 2 * 4 * 16 = 128

constexpr size_t TILE_K = 32;
constexpr size_t PAD = 8;

// A[BLOCK_TILE_ROWS][TILE_K]
constexpr size_t A_THREADS_PER_ROW = TILE_K / 8;                           // 32 / 8 = 4
constexpr size_t A_ROWS_PER_STEP = NUM_THREADS / A_THREADS_PER_ROW;        // 256 / 4 = 64
constexpr size_t A_STEPS = BLOCK_TILE_ROWS / A_ROWS_PER_STEP;              // 128 / 64 = 2

// B[TILE_K][BLOCK_TILE_COLS]
constexpr size_t B_THREADS_PER_ROW = BLOCK_TILE_COLS / 8;                  // 128 / 8 = 16
constexpr size_t B_ROWS_PER_STEP = NUM_THREADS / B_THREADS_PER_ROW;        // 256 / 16 = 16
constexpr size_t B_STEPS = TILE_K / B_ROWS_PER_STEP;                       // 32 / 16 = 2

constexpr size_t A_FP64_PER_ROW = TILE_K / 4;
constexpr size_t A_NOTALIGNED_STEPS = (BLOCK_TILE_ROWS * A_FP64_PER_ROW) / NUM_THREADS;
constexpr size_t B_FP64_PER_ROW = BLOCK_TILE_COLS / 4;
constexpr size_t B_NOTALIGNED_STEPS = (TILE_K * B_FP64_PER_ROW) / NUM_THREADS;

template <bool IS_ALIGNED>
static __global__ void _gemm_nkm_wmma_simple(
    fp16 *A,
    fp16 *B,
    fp32 *C,
    size_t N, size_t K, size_t M
) {
    size_t threadId = threadIdx.y * blockDim.x + threadIdx.x;
    size_t warpId = threadId / threadsPerWarp;
    size_t warpRow = warpId / WARP_BLOCK_COLS;
    size_t warpCol = warpId % WARP_BLOCK_COLS;

    size_t globalBlockRow = blockIdx.y * BLOCK_TILE_ROWS;
    size_t globalBlockCol = blockIdx.x * BLOCK_TILE_COLS;

    __shared__ fp16 block_A[2][BLOCK_TILE_ROWS][TILE_K + PAD];
    __shared__ fp16 block_B[2][TILE_K][BLOCK_TILE_COLS + PAD];

    wmma::fragment<wmma::matrix_a, tileSize, tileSize, tileSize, fp16, wmma::row_major> a_frag[WARP_TILE_ROWS];
    wmma::fragment<wmma::matrix_b, tileSize, tileSize, tileSize, fp16, wmma::row_major> b_frag[WARP_TILE_COLS];
    wmma::fragment<wmma::accumulator, tileSize, tileSize, tileSize, fp32> c_frag[WARP_TILE_ROWS][WARP_TILE_COLS];

    for (int i = 0; i < WARP_TILE_ROWS; i++) {
        for (int j = 0; j < WARP_TILE_COLS; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Prologue
    if constexpr (IS_ALIGNED) {
        for (int step = 0; step < A_STEPS; step++) {
            int r_a = step * A_ROWS_PER_STEP + threadId / A_THREADS_PER_ROW;
            int c_a = (threadId % A_THREADS_PER_ROW) * 8;
            __pipeline_memcpy_async(&block_A[0][r_a][c_a], &A[(globalBlockRow + r_a) * K + c_a], 16);
        }
        for (int step = 0; step < B_STEPS; step++) {
            int r_b = step * B_ROWS_PER_STEP + threadId / B_THREADS_PER_ROW;
            int c_b = (threadId % B_THREADS_PER_ROW) * 8;
            __pipeline_memcpy_async(&block_B[0][r_b][c_b], &B[r_b * M + (globalBlockCol + c_b)], 16);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
    } else {
        for (int step = 0; step < A_NOTALIGNED_STEPS; step++) {
            int idx = step * NUM_THREADS + threadId;
            int r_a = idx / A_FP64_PER_ROW;
            int c_a = (idx % A_FP64_PER_ROW) * 4;
            *(reinterpret_cast<fp64*>(&block_A[0][r_a][c_a])) = safe_load<false>(A, globalBlockRow + r_a, c_a, N, K, K);
        }
        for (int step = 0; step < B_NOTALIGNED_STEPS; step++) {
            int idx = step * NUM_THREADS + threadId;
            int r_b = idx / B_FP64_PER_ROW;
            int c_b = (idx % B_FP64_PER_ROW) * 4;
            *(reinterpret_cast<fp64*>(&block_B[0][r_b][c_b])) = safe_load<false>(B, r_b, globalBlockCol + c_b, K, M, M);
        }
    }
    __syncthreads();


    // main for
    for (size_t i = 0; i < K; i += TILE_K) {
        int current = (i / TILE_K) % 2;
        int next = (current + 1) % 2;
        size_t next_i = i + TILE_K;

        if (next_i < K) {
            if constexpr (IS_ALIGNED) {
                for (int step = 0; step < A_STEPS; step++) {
                    int r_a = step * A_ROWS_PER_STEP + threadId / A_THREADS_PER_ROW;
                    int c_a = (threadId % A_THREADS_PER_ROW) * 8;
                    __pipeline_memcpy_async(&block_A[next][r_a][c_a], &A[(globalBlockRow + r_a) * K + (next_i + c_a)], 16);
                }
                for (int step = 0; step < B_STEPS; step++) {
                    int r_b = step * B_ROWS_PER_STEP + threadId / B_THREADS_PER_ROW;
                    int c_b = (threadId % B_THREADS_PER_ROW) * 8;
                    __pipeline_memcpy_async(&block_B[next][r_b][c_b], &B[(next_i + r_b) * M + (globalBlockCol + c_b)], 16);
                }
                __pipeline_commit();
            } else {
                for (int step = 0; step < A_NOTALIGNED_STEPS; step++) {
                    int idx = step * NUM_THREADS + threadId;
                    int r_a = idx / A_FP64_PER_ROW;
                    int c_a = (idx % A_FP64_PER_ROW) * 4;
                    *(reinterpret_cast<fp64*>(&block_A[next][r_a][c_a])) = safe_load<false>(A, globalBlockRow + r_a, next_i + c_a, N, K, K);
                }
                for (int step = 0; step < B_NOTALIGNED_STEPS; step++) {
                    int idx = step * NUM_THREADS + threadId;
                    int r_b = idx / B_FP64_PER_ROW;
                    int c_b = (idx % B_FP64_PER_ROW) * 4;
                    *(reinterpret_cast<fp64*>(&block_B[next][r_b][c_b])) = safe_load<false>(B, next_i + r_b, globalBlockCol + c_b, K, M, M);
                }
            }
        }

        constexpr size_t NUM_K_STEPS = TILE_K / tileSize; // 32 / 16 = 2
        for (int k_step = 0; k_step < NUM_K_STEPS; ++k_step) {
            for (int m = 0; m < WARP_TILE_ROWS; m++) {
                wmma::load_matrix_sync(a_frag[m], &block_A[current][warpRow * (WARP_TILE_ROWS * tileSize) + m * tileSize][k_step * tileSize], TILE_K + PAD);
            }
            for (int n = 0; n < WARP_TILE_COLS; n++) {
                wmma::load_matrix_sync(b_frag[n], &block_B[current][k_step * tileSize][warpCol * (WARP_TILE_COLS * tileSize) + n * tileSize], BLOCK_TILE_COLS + PAD);
            }
            for (int m = 0; m < WARP_TILE_ROWS; m++) {
                for (int n = 0; n < WARP_TILE_COLS; n++) {
                    wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
                }
            }
        }

        if (next_i < K) {
            if constexpr (IS_ALIGNED) {
                __pipeline_wait_prior(0);
            }
        }
        __syncthreads();
    }

    if constexpr (IS_ALIGNED) {
        for (int m = 0; m < WARP_TILE_ROWS; m++) {
            for (int n = 0; n < WARP_TILE_COLS; n++) {
                size_t g_r = globalBlockRow + warpRow * (WARP_TILE_ROWS * tileSize) + m * tileSize;
                size_t g_c = globalBlockCol + warpCol * (WARP_TILE_COLS * tileSize) + n * tileSize;
                wmma::store_matrix_sync(C + g_r * M + g_c, c_frag[m][n], M, wmma::mem_row_major);
            }
        }
    } else {
        float* warp_buf = reinterpret_cast<float*>(block_A);
        constexpr size_t WARP_C_ELEMENTS = (WARP_TILE_ROWS * tileSize) * (WARP_TILE_COLS * tileSize); // 32 * 64 = 2048
        for (int w = 0; w < NUM_WARPS; w++) {
            if (warpId == w) {
                for (int m = 0; m < WARP_TILE_ROWS; m++) {
                    for (int n = 0; n < WARP_TILE_COLS; n++) {
                        wmma::store_matrix_sync(&warp_buf[(m * tileSize) * (WARP_TILE_COLS * tileSize) + (n * tileSize)], c_frag[m][n], WARP_TILE_COLS * tileSize, wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            for (int idx = threadId; idx < WARP_C_ELEMENTS; idx += NUM_THREADS) {
                int l_r = idx / (WARP_TILE_COLS * tileSize);
                int l_c = idx % (WARP_TILE_COLS * tileSize);

                int w_r = w / WARP_BLOCK_COLS;
                int w_c = w % WARP_BLOCK_COLS;

                size_t g_r = globalBlockRow + w_r * (WARP_TILE_ROWS * tileSize) + l_r;
                size_t g_c = globalBlockCol + w_c * (WARP_TILE_COLS * tileSize) + l_c;

                if (g_r < N && g_c < M) {
                    C[g_r * M + g_c] = warp_buf[idx];
                }
            }
            __syncthreads();
        }
    }
}

__host__ void _gemm_nkm_wmma_launcher(Matrix<fp16> &A, Matrix<fp16> &B, Matrix<fp32> &C) {
    size_t N = A.shape().first;
    size_t K = A.shape().second;
    size_t M = B.shape().second;

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    bool is_aligned = (K % 16 == 0) && (N % BLOCK_TILE_ROWS == 0) && (M % BLOCK_TILE_COLS == 0);

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim((M + BLOCK_TILE_COLS - 1) / BLOCK_TILE_COLS,
                  (N + BLOCK_TILE_ROWS - 1) / BLOCK_TILE_ROWS);

    if (is_aligned) {
        _gemm_nkm_wmma_simple<true><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
    } else {
        _gemm_nkm_wmma_simple<false><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item(), N, K, M);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}