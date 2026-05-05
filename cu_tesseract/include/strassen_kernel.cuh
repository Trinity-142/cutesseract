#pragma once

#include <cuda_runtime.h>

#include "dtypes.cuh"
#include "matrix.cuh"
#include "utils.cuh"
#include "kernels.cuh"


template <typename T>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C);
 
#define CUTOFF_SIZE 256
template <typename T>
__host__ void _gemm_strassen_launcher(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
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

    _gemm_strassen<T>(A, B, C);
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


template <typename T>
__global__ void add_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * ldc + col] = A[row * lda + col] + B[row * ldb + col];
    }
}

template <typename T>
__global__ void sub_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * ldc + col] = A[row * lda + col] - B[row * ldb + col];
    }
}

template <typename T>
__global__ void gemm_base_square_kernel(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc, size_t N) {
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



template <typename T>
__host__ void _strassen_rec(const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc, size_t N,
                            cudaStream_t stream = 0) {
    if (N <= CUTOFF_SIZE) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                      (N + block_dim.y - 1) / block_dim.y);

        gemm_base_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(A, lda, B, ldb, C, ldc, N);
        return;
    } else {
        size_t H = N / 2;

        const T* A11 = A;
        const T* A12 = A + H;
        const T* A21 = A + H * lda;
        const T* A22 = A + H * lda + H;

        const T* B11 = B;
        const T* B12 = B + H;
        const T* B21 = B + H * ldb;
        const T* B22 = B + H * ldb + H;

        T* C11 = C;
        T* C12 = C + H;
        T* C21 = C + H * ldc;
        T* C22 = C + H * ldc + H;

        cudaStream_t s[7];
        for (int i = 0; i < 7; i++) {
            CUDA_CHECK(cudaStreamCreate(&s[i]));
        }

        Matrix<T> S1_P1(H, H, ROW_WISE, CUDA), S2_P1(H, H, ROW_WISE, CUDA);
        Matrix<T> S1_P2(H, H, ROW_WISE, CUDA);
        Matrix<T> S2_P3(H, H, ROW_WISE, CUDA);
        Matrix<T> S2_P4(H, H, ROW_WISE, CUDA);
        Matrix<T> S1_P5(H, H, ROW_WISE, CUDA);
        Matrix<T> S1_P6(H, H, ROW_WISE, CUDA), S2_P6(H, H, ROW_WISE, CUDA);
        Matrix<T> S1_P7(H, H, ROW_WISE, CUDA), S2_P7(H, H, ROW_WISE, CUDA);

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

        // P1 = (A11 + A22) * (B11 + B22)
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[0]>>>(A11, lda, A22, lda, S1_P1.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[0]>>>(B11, ldb, B22, ldb, S2_P1.item(), H, H);
        _strassen_rec<T>(S1_P1.item(), H, S2_P1.item(), H, P1.item(), H, H, s[0]);

        // P2 = (A21 + A22) * B11
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[1]>>>(A21, lda, A22, lda, S1_P2.item(), H, H);
        _strassen_rec<T>(S1_P2.item(), H, B11, ldb, P2.item(), H, H, s[1]);

        // P3 = A11 * (B12 - B22)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[2]>>>(B12, ldb, B22, ldb, S2_P3.item(), H, H);
        _strassen_rec<T>(A11, lda, S2_P3.item(), H, P3.item(), H, H, s[2]);

        // P4 = A22 * (B21 - B11)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[3]>>>(B21, ldb, B11, ldb, S2_P4.item(), H, H);
        _strassen_rec<T>(A22, lda, S2_P4.item(), H, P4.item(), H, H, s[3]);

        // P5 = (A11 + A12) * B22
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[4]>>>(A11, lda, A12, lda, S1_P5.item(), H, H);
        _strassen_rec<T>(S1_P5.item(), H, B22, ldb, P5.item(), H, H, s[4]);

        // P6 = (A21 - A11) * (B11 + B12)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[5]>>>(A21, lda, A11, lda, S1_P6.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[5]>>>(B11, ldb, B12, ldb, S2_P6.item(), H, H);
        _strassen_rec<T>(S1_P6.item(), H, S2_P6.item(), H, P6.item(), H, H, s[5]);

        // P7 = (A12 - A22) * (B21 + B22)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[6]>>>(A12, lda, A22, lda, S1_P7.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[6]>>>(B21, ldb, B22, ldb, S2_P7.item(), H, H);
        _strassen_rec<T>(S1_P7.item(), H, S2_P7.item(), H, P7.item(), H, H, s[6]);

        for (int i = 0; i < 7; i++) {
            CUDA_CHECK(cudaStreamSynchronize(s[i]));
            CUDA_CHECK(cudaStreamDestroy(s[i]));
        }

        Matrix<T> T1(H, H, ROW_WISE, CUDA);
        Matrix<T> T2(H, H, ROW_WISE, CUDA);

        // C11 = P1 + P4 - P5 + P7
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P1.item(), H, P4.item(), H, T1.item(), H, H);
        sub_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T1.item(), H, P5.item(), H, T2.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T2.item(), H, P7.item(), H, C11, ldc, H);

        // C12 = P3 + P5
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P3.item(), H, P5.item(), H, C12, ldc, H);

        // C21 = P2 + P4
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P2.item(), H, P4.item(), H, C21, ldc, H);

        // C22 = P1 - P2 + P3 + P6
        sub_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P1.item(), H, P2.item(), H, T1.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T1.item(), H, P3.item(), H, T2.item(), H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T2.item(), H, P6.item(), H, C22, ldc, H);

        return;
    }
}



template <typename T>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
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

    size_t S = next_pow2(max3(N, K, M)); // typa kvadrat

    // could be optimizable, do not care enough
    if (S <= CUTOFF_SIZE) {
        _gemm_nkm_simple_launcher<T>(A, B, C);
        return;
    }

    if (N == K && K == M && (N == S)) {
        _strassen_rec<T>(A.item(), N, B.item(), K, C.item(), M, S);
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

    _strassen_rec<T>(Ap.item(), S, Bp.item(), S, Cp.item(), S, S);

    // Copy back only the valid N x M result.
    copy_rect_kernel<T><<<gridC, block_dim>>>(Cp.item(), S, C.item(), M, N, M);
    CUDA_CHECK(cudaDeviceSynchronize());
}
