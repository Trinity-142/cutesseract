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

template <size_t N>
static __global__ void _gemm_nn_wmma_simple(
    half *A,
    half *B,
    fp32 *C
) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  size_t col = blockIdx.x * WMMA_N;
  size_t row = blockIdx.y * WMMA_M;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (size_t i = 0; i < N; i += WMMA_K) {
    wmma::load_matrix_sync(a_frag, A + row * N + i, N);
    wmma::load_matrix_sync(b_frag, B + i * N + col, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

template <size_t N>
__host__ void _gemm_nn_wmma_launcher(Matrix<half> &A, Matrix<half> &B, Matrix<fp32> &C) {
  assert(A.shape().first == N && A.shape().second == N);
  assert(B.shape().first == N && B.shape().second == N);
  assert(C.shape().first == N && C.shape().second == N);

  assert((N % 16) == 0);

  assert(A.get_layout() == ROW_WISE);
  assert(B.get_layout() == ROW_WISE);
  assert(C.get_layout() == ROW_WISE);

  A.cuda();
  B.cuda();
  C.cuda();

  dim3 block_dim(32, 1);
  dim3 grid_dim(N / 16, N / 16);

  //cudaFuncSetCacheConfig(_gemm_nn_wmma_launcher<N>, cudaFuncCachePreferShared);
  _gemm_nn_wmma_simple<N><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
  CUDA_CHECK(cudaDeviceSynchronize());
}
