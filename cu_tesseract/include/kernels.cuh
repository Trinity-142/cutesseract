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


template <typename T, size_t N, size_t BS>
__host__ void _gemm_nn_block_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
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

    cudaFuncSetCacheConfig(_gemm_nnn_block_simple<N, BS>, cudaFuncCachePreferShared);
    _gemm_nnn_block_simple<N, BS><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T, size_t N, size_t BS>
static __global__ void _gemm_nnn_block_simple(
    T *A, // row-wise
    T *B, // row-wise
    T *C // row-wise
) {
    assert(blockDim.x == BS && blockDim.y == BS);

    size_t row = blockIdx.y * BS + threadIdx.y;
    size_t col = blockIdx.x * BS + threadIdx.x;

    T sum = 0.0;

    __shared__ T block_a[BS][BS];
    __shared__ T block_b[BS][BS];

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


template <typename T, size_t N, size_t K, size_t M> // n*k x k*m = n*m
__host__ void _gemm_nkm_simple_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    assert(A.shape().first == N && A.shape().second == K);
    assert(B.shape().first == K && B.shape().second == M);
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

    cudaFuncSetCacheConfig(_gemm_nkm_simple<N, K, M>, cudaFuncCachePreferL1);
    _gemm_nkm_simple<N, K, M><<<grid_dim, block_dim>>>(A.item(), B.item(), C.item());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template <typename T, size_t N, size_t K, size_t M> // n*k x k*m = n*m
static __global__ void _gemm_nkm_simple(
    T *A, // row-wise
    T *B, // row-wise
    T *C // row-wise
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k*M + col];
        }
        C[row * M + col] = sum;
    }
}


#define CUTOFF_SIZE 3072
template <typename T, size_t N, size_t K, size_t M>
__host__ void _gemm_strassen_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    assert(A.shape().first == N && A.shape().second == K);
    assert(B.shape().first == K && B.shape().second == M);
    assert(C.shape().first == N && C.shape().second == M);

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    _gemm_strassen<N, K, M>(A, B, C);
}

template <typename T>//                                   leading destination
__global__ void copy_rect_kernel(const T* src, size_t src_ld, T* dst, size_t dst_ld, size_t rows, size_t cols) {
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
    while (p < x) p <<= 1;
    return p;
}


template <typename T, size_t N>
__host__ void _strassen_rec(const T* A, size_t lda,
                            const T* B, size_t ldb,
                            T* C, size_t ldc) {
    if constexpr (N <= CUTOFF_SIZE) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                      (N + block_dim.y - 1) / block_dim.y);

        gemm_base_square_kernel<N><<<grid_dim, block_dim>>>(A, lda, B, ldb, C, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    } else {
        constexpr size_t H = N / 2;

        // Quadrant pointers for row-wise layout.
        const fp32* A11 = A;
        const fp32* A12 = A + H;
        const fp32* A21 = A + H * lda;
        const fp32* A22 = A + H * lda + H;

        const fp32* B11 = B;
        const fp32* B12 = B + H;
        const fp32* B21 = B + H * ldb;
        const fp32* B22 = B + H * ldb + H;

        fp32* C11 = C;
        fp32* C12 = C + H;
        fp32* C21 = C + H * ldc;
        fp32* C22 = C + H * ldc + H;

        // Scratch matrices.
        Matrix<T> S1(H, H, ROW_WISE, CUDA);
        Matrix<T> S2(H, H, ROW_WISE, CUDA);
        Matrix<T> T1(H, H, ROW_WISE, CUDA);
        Matrix<T> T2(H, H, ROW_WISE, CUDA);

        Matrix<T> P1(H, H, ROW_WISE, CUDA);
        Matrix<T> P2(H, H, ROW_WISE, CUDA);
        Matrix<T> P3(H, H, ROW_WISE, CUDA);
        Matrix<T> P4(H, H, ROW_WISE, CUDA);
        Matrix<T> P5(H, H, ROW_WISE, CUDA);
        Matrix<T> P6(H, H, ROW_WISE, CUDA);
        Matrix<T> P7(H, H, ROW_WISE, CUDA);

        dim3 block_dim(16, 16);
        dim3 grid_dim((H + block_dim.x - 1) / block_dim.x,
                      (H + block_dim.y - 1) / block_dim.y);

        // M1 = (A11 + A22) * (B11 + B22)
        add_square_kernel<H><<<grid_dim, block_dim>>>(A11, lda, A22, lda, S1.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(B11, ldb, B22, ldb, S2.item(), H);
        _strassen_rec<H>(S1.item(), H, S2.item(), H, P1.item(), H);

        // M2 = (A21 + A22) * B11
        add_square_kernel<H><<<grid_dim, block_dim>>>(A21, lda, A22, lda, S1.item(), H);
        _strassen_rec<H>(S1.item(), H, B11, ldb, P2.item(), H);

        // M3 = A11 * (B12 - B22)
        sub_square_kernel<H><<<grid_dim, block_dim>>>(B12, ldb, B22, ldb, S2.item(), H);
        _strassen_rec<H>(A11, lda, S2.item(), H, P3.item(), H);

        // M4 = A22 * (B21 - B11)
        sub_square_kernel<H><<<grid_dim, block_dim>>>(B21, ldb, B11, ldb, S2.item(), H);
        _strassen_rec<H>(A22, lda, S2.item(), H, P4.item(), H);

        // M5 = (A11 + A12) * B22
        add_square_kernel<H><<<grid_dim, block_dim>>>(A11, lda, A12, lda, S1.item(), H);
        _strassen_rec<H>(S1.item(), H, B22, ldb, P5.item(), H);

        // M6 = (A21 - A11) * (B11 + B12)
        sub_square_kernel<H><<<grid_dim, block_dim>>>(A21, lda, A11, lda, S1.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(B11, ldb, B12, ldb, S2.item(), H);
        _strassen_rec<H>(S1.item(), H, S2.item(), H, P6.item(), H);

        // M7 = (A12 - A22) * (B21 + B22)
        sub_square_kernel<H><<<grid_dim, block_dim>>>(A12, lda, A22, lda, S1.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(B21, ldb, B22, ldb, S2.item(), H);
        _strassen_rec<H>(S1.item(), H, S2.item(), H, P7.item(), H);

        // Combine:
        // C11 = P1 + P4 - P5 + P7
        add_square_kernel<H><<<grid_dim, block_dim>>>(P1.item(), H, P4.item(), H, T1.item(), H);
        sub_square_kernel<H><<<grid_dim, block_dim>>>(T1.item(), H, P5.item(), H, T2.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(T2.item(), H, P7.item(), H, C11, ldc);

        // C12 = P3 + P5
        add_square_kernel<H><<<grid_dim, block_dim>>>(P3.item(), H, P5.item(), H, C12, ldc);

        // C21 = P2 + P4
        add_square_kernel<H><<<grid_dim, block_dim>>>(P2.item(), H, P4.item(), H, C21, ldc);

        // C22 = P1 - P2 + P3 + P6
        sub_square_kernel<H><<<grid_dim, block_dim>>>(P1.item(), H, P2.item(), H, T1.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(T1.item(), H, P3.item(), H, T2.item(), H);
        add_square_kernel<H><<<grid_dim, block_dim>>>(T2.item(), H, P6.item(), H, C22, ldc);

        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
}



template <typename T, size_t N, size_t K, size_t M>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    assert(A.shape().first == N && A.shape().second == K);
    assert(B.shape().first == K && B.shape().second == M);
    assert(C.shape().first == N && C.shape().second == M);

    assert(A.get_layout() == ROW_WISE);
    assert(B.get_layout() == ROW_WISE);
    assert(C.get_layout() == ROW_WISE);

    A.cuda();
    B.cuda();
    C.cuda();

    constexpr size_t S = next_pow2(max3(N, K, M)); // typa kvadrat

    // could be optimizable, do not care enough
    if constexpr (S <= CUTOFF_SIZE) {
        _gemm_nkm_simple_launcher<N, K, M>(A, B, C);
        return;
    }

    if constexpr (N == K && K == M && (N == S)) {
        _strassen_rec<S>(A.item(), N, B.item(), K, C.item(), M);
        return;
    }

    Matrix<T> Ap(S, S, ROW_WISE, CUDA);
    Matrix<T> Bp(S, S, ROW_WISE, CUDA);
    Matrix<T> Cp(S, S, ROW_WISE, CUDA);

    CUDA_CHECK(cudaMemset(Ap.item(), 0, sizeof(T) * S * S));
    CUDA_CHECK(cudaMemset(Bp.item(), 0, sizeof(T) * S * S));
    CUDA_CHECK(cudaMemset(Cp.item(), 0, sizeof(T) * S * S));

    dim3 block_dim(16, 16);
    dim3 gridA((K + block_dim.x - 1) / block_dim.x,
               (N + block_dim.y - 1) / block_dim.y);
    dim3 gridB((M + block_dim.x - 1) / block_dim.x,
               (K + block_dim.y - 1) / block_dim.y);
    dim3 gridC((M + block_dim.x - 1) / block_dim.x,
               (N + block_dim.y - 1) / block_dim.y);

    copy_rect_kernel<T><<<gridA, block_dim>>>(A.item(), K, Ap.item(), S, N, K);
    copy_rect_kernel<T><<<gridB, block_dim>>>(B.item(), M, Bp.item(), S, K, M);

    CUDA_CHECK(cudaDeviceSynchronize());

    _strassen_rec<S>(Ap.item(), S, Bp.item(), S, Cp.item(), S);

    // Copy back only the valid N x M result.
    copy_rect_kernel<T><<<gridC, block_dim>>>(Cp.item(), S, C.item(), M, N, M);
    CUDA_CHECK(cudaDeviceSynchronize());
}
