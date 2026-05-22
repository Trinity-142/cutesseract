#pragma once

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

#include "dtypes.cuh"
#include "kernels.cuh"
#include "matrix.cuh"
#include "utils.cuh"

template <typename T>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C);

#ifndef CUTOFF_SIZE
#define CUTOFF_SIZE 256
#endif

#ifndef STRASSEN_PARALLEL_LEVELS
#define STRASSEN_PARALLEL_LEVELS 1
#endif

template <typename T>
size_t get_strassen_workspace_size(size_t N, int depth = 0) {
  if (N <= CUTOFF_SIZE)
    return 0;
  size_t H = N / 2;
  size_t child_size = get_strassen_workspace_size<T>(H, depth + 1);

  if (depth < STRASSEN_PARALLEL_LEVELS) {
    return 19 * H * H + 7 * child_size;
  } else {
    return 19 * H * H + 1 * child_size;
  }
}

inline size_t count_total_strassen_streams(int levels) {
  size_t total = 0;
  size_t current_level_width = 1;
  for (int i = 0; i < levels; ++i) {
    current_level_width *= 7;
    total += current_level_width;
  }
  return total;
}

template <typename T> //                                   leading destination
__global__ void copy_rect_kernel(const T *src, size_t src_ld, T *dst,
                                 size_t dst_ld, size_t rows, size_t cols) {
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < rows && col < cols) {
    dst[row * dst_ld + col] = src[row * src_ld + col];
  }
}

constexpr size_t max3(size_t a, size_t b, size_t c) {
  return std::max(a, std::max(b, c));
}

constexpr size_t next_pow2(size_t x) {
  size_t p = 1;
  while (p < x)
    p <<= 1;
  return p;
}

template <typename T>
__global__ void add_square_kernel(const T *A, size_t lda, const T *B,
                                  size_t ldb, T *C, size_t ldc, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    C[row * ldc + col] = A[row * lda + col] + B[row * ldb + col];
  }
}

template <typename T>
__global__ void sub_square_kernel(const T *A, size_t lda, const T *B,
                                  size_t ldb, T *C, size_t ldc, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    C[row * ldc + col] = A[row * lda + col] - B[row * ldb + col];
  }
}

template <typename T>
__host__ void _strassen_rec(const T *A, size_t lda, const T *B, size_t ldb,
                            T *C, size_t ldc, size_t N, T *workspace, int depth,
                            cudaStream_t *stream_pool, size_t &stream_offset,
                            cudaStream_t current_stream) {
  if (N <= CUTOFF_SIZE) {
    size_t BS = 16;
    dim3 block_dim(BS, BS);
    dim3 grid_dim((N + BS - 1) / BS, (N + BS - 1) / BS);
    size_t shared_mem_size = 2 * BS * BS * sizeof(T);

    // Using optimized blocked GEMM definition from kernels.cuh
    _gemm_nnn_block_simple<T>
        <<<grid_dim, block_dim, shared_mem_size, current_stream>>>(
            const_cast<T *>(A), const_cast<T *>(B), C, N, BS);
    return;
  }

  size_t H = N / 2;
  size_t H2 = H * H;

  const T *A11 = A;
  const T *A12 = A + H;
  const T *A21 = A + H * lda;
  const T *A22 = A + H * lda + H;

  const T *B11 = B;
  const T *B12 = B + H;
  const T *B21 = B + H * ldb;
  const T *B22 = B + H * ldb + H;

  T *C11 = C;
  T *C12 = C + H;
  T *C21 = C + H * ldc;
  T *C22 = C + H * ldc + H;

  T *S1_P1 = workspace;
  T *S2_P1 = workspace + H2;
  T *S1_P2 = workspace + 2 * H2;
  T *S2_P3 = workspace + 3 * H2;
  T *S2_P4 = workspace + 4 * H2;
  T *S1_P5 = workspace + 5 * H2;
  T *S1_P6 = workspace + 6 * H2;
  T *S2_P6 = workspace + 7 * H2;
  T *S1_P7 = workspace + 8 * H2;
  T *S2_P7 = workspace + 9 * H2;

  T *P1 = workspace + 10 * H2;
  T *P2 = workspace + 11 * H2;
  T *P3 = workspace + 12 * H2;
  T *P4 = workspace + 13 * H2;
  T *P5 = workspace + 14 * H2;
  T *P6 = workspace + 15 * H2;
  T *P7 = workspace + 16 * H2;

  T *T1 = workspace + 17 * H2;
  T *T2 = workspace + 18 * H2;

  T *next_ws = workspace + 19 * H2;
  size_t child_ws_size = get_strassen_workspace_size<T>(H, depth + 1);

  dim3 block_dim(16, 16);
  dim3 grid_dim((H + block_dim.x - 1) / block_dim.x,
                (H + block_dim.y - 1) / block_dim.y);

  if (depth < STRASSEN_PARALLEL_LEVELS) {
    cudaStream_t s[7];
    for (int i = 0; i < 7; i++)
      s[i] = stream_pool[stream_offset++];

    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[0]>>>(A11, lda, A22, lda, S1_P1, H, H);
    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[0]>>>(B11, ldb, B22, ldb, S2_P1, H, H);
    _strassen_rec<T>(S1_P1, H, S2_P1, H, P1, H, H, next_ws + 0 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[0]);

    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[1]>>>(A21, lda, A22, lda, S1_P2, H, H);
    _strassen_rec<T>(S1_P2, H, B11, ldb, P2, H, H, next_ws + 1 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[1]);

    sub_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[2]>>>(B12, ldb, B22, ldb, S2_P3, H, H);
    _strassen_rec<T>(A11, lda, S2_P3, H, P3, H, H, next_ws + 2 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[2]);

    sub_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[3]>>>(B21, ldb, B11, ldb, S2_P4, H, H);
    _strassen_rec<T>(A22, lda, S2_P4, H, P4, H, H, next_ws + 3 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[3]);

    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[4]>>>(A11, lda, A12, lda, S1_P5, H, H);
    _strassen_rec<T>(S1_P5, H, B22, ldb, P5, H, H, next_ws + 4 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[4]);

    sub_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[5]>>>(A21, lda, A11, lda, S1_P6, H, H);
    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[5]>>>(B11, ldb, B12, ldb, S2_P6, H, H);
    _strassen_rec<T>(S1_P6, H, S2_P6, H, P6, H, H, next_ws + 5 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[5]);

    sub_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[6]>>>(A12, lda, A22, lda, S1_P7, H, H);
    add_square_kernel<T>
        <<<grid_dim, block_dim, 0, s[6]>>>(B21, ldb, B22, ldb, S2_P7, H, H);
    _strassen_rec<T>(S1_P7, H, S2_P7, H, P7, H, H, next_ws + 6 * child_ws_size,
                     depth + 1, stream_pool, stream_offset, s[6]);

    for (int i = 0; i < 7; i++)
      CUDA_CHECK(cudaStreamSynchronize(s[i]));
  } else {
    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        A11, lda, A22, lda, S1_P1, H, H);
    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        B11, ldb, B22, ldb, S2_P1, H, H);
    _strassen_rec<T>(S1_P1, H, S2_P1, H, P1, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        A21, lda, A22, lda, S1_P2, H, H);
    _strassen_rec<T>(S1_P2, H, B11, ldb, P2, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    sub_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        B12, ldb, B22, ldb, S2_P3, H, H);
    _strassen_rec<T>(A11, lda, S2_P3, H, P3, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    sub_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        B21, ldb, B11, ldb, S2_P4, H, H);
    _strassen_rec<T>(A22, lda, S2_P4, H, P4, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        A11, lda, A12, lda, S1_P5, H, H);
    _strassen_rec<T>(S1_P5, H, B22, ldb, P5, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    sub_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        A21, lda, A11, lda, S1_P6, H, H);
    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        B11, ldb, B12, ldb, S2_P6, H, H);
    _strassen_rec<T>(S1_P6, H, S2_P6, H, P6, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);

    sub_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        A12, lda, A22, lda, S1_P7, H, H);
    add_square_kernel<T><<<grid_dim, block_dim, 0, current_stream>>>(
        B21, ldb, B22, ldb, S2_P7, H, H);
    _strassen_rec<T>(S1_P7, H, S2_P7, H, P7, H, H, next_ws, depth + 1,
                     stream_pool, stream_offset, current_stream);
  }

  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(P1, H, P4, H, T1, H, H);
  sub_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(T1, H, P5, H, T2, H, H);
  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(T2, H, P7, H, C11, ldc, H);
  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(P3, H, P5, H, C12, ldc, H);
  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(P2, H, P4, H, C21, ldc, H);
  sub_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(P1, H, P2, H, T1, H, H);
  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(T1, H, P3, H, T2, H, H);
  add_square_kernel<T>
      <<<grid_dim, block_dim, 0, current_stream>>>(T2, H, P6, H, C22, ldc, H);
}

template <typename T>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
  size_t N = A.shape().first;
  size_t K = A.shape().second;
  size_t M = B.shape().second;

  size_t S = next_pow2(max3(N, K, M));

  if (S <= CUTOFF_SIZE) {
    _gemm_nn_block_launcher<T>(A, B, C);
    return;
  }

  size_t pad_size = 3 * S * S;
  size_t ws_size = get_strassen_workspace_size<T>(S);
  T *d_memory = nullptr;
  CUDA_CHECK(cudaMalloc(&d_memory, (pad_size + ws_size) * sizeof(T)));
  CUDA_CHECK(cudaMemset(d_memory, 0, (pad_size + ws_size) * sizeof(T)));

  T *Ap = d_memory, *Bp = d_memory + S * S, *Cp = d_memory + 2 * S * S;
  T *workspace = d_memory + 3 * S * S;

  size_t num_streams = count_total_strassen_streams(STRASSEN_PARALLEL_LEVELS);
  cudaStream_t *stream_pool = new cudaStream_t[num_streams];
  for (size_t i = 0; i < num_streams; i++)
    CUDA_CHECK(cudaStreamCreate(&stream_pool[i]));

  dim3 block(16, 16);
  copy_rect_kernel<<<dim3((K + 15) / 16, (N + 15) / 16), block>>>(A.item(), K,
                                                                  Ap, S, N, K);
  copy_rect_kernel<<<dim3((M + 15) / 16, (K + 15) / 16), block>>>(B.item(), M,
                                                                  Bp, S, K, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  size_t stream_offset = 0;
  _strassen_rec<T>(Ap, S, Bp, S, Cp, S, S, workspace, 0, stream_pool,
                   stream_offset, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  copy_rect_kernel<<<dim3((M + 15) / 16, (N + 15) / 16), block>>>(
      Cp, S, C.item(), M, N, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (size_t i = 0; i < num_streams; i++)
    CUDA_CHECK(cudaStreamDestroy(stream_pool[i]));
  delete[] stream_pool;
  CUDA_CHECK(cudaFree(d_memory));
}

template <typename T>
__host__ void _gemm_strassen_launcher(Matrix<T> &A, Matrix<T> &B,
                                      Matrix<T> &C) {
  A.cuda();
  B.cuda();
  C.cuda();
  _gemm_strassen<T>(A, B, C);
}
