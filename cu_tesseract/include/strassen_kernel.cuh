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
size_t get_strassen_workspace_size(size_t N) {
    if (N <= CUTOFF_SIZE) return 0;
    size_t H = N / 2;
    // 19 matrices of size HxH at this level, plus space for 7 recursive calls
    return 19 * H * H + 7 * get_strassen_workspace_size<T>(H);
}

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
__host__ void _strassen_rec(const T* A, size_t lda,
                            const T* B, size_t ldb,
                            T* C, size_t ldc,
                            size_t N,
                            T* workspace,
                            cudaStream_t stream = 0) {
    if (N <= CUTOFF_SIZE) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                      (N + block_dim.y - 1) / block_dim.y);

        gemm_base_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(A, lda, B, ldb, C, ldc, N);
        return;
    } else {
        size_t H = N / 2;
        size_t H2 = H * H;

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

        // Partition workspace for this level
        // 10 scratch inputs + 7 products + 2 final scratch = 19
        T* S1_P1 = workspace;
        T* S2_P1 = workspace + H2;
        T* S1_P2 = workspace + 2 * H2;
        T* S2_P3 = workspace + 3 * H2;
        T* S2_P4 = workspace + 4 * H2;
        T* S1_P5 = workspace + 5 * H2;
        T* S1_P6 = workspace + 6 * H2;
        T* S2_P6 = workspace + 7 * H2;
        T* S1_P7 = workspace + 8 * H2;
        T* S2_P7 = workspace + 9 * H2;

        T* P1 = workspace + 10 * H2;
        T* P2 = workspace + 11 * H2;
        T* P3 = workspace + 12 * H2;
        T* P4 = workspace + 13 * H2;
        T* P5 = workspace + 14 * H2;
        T* P6 = workspace + 15 * H2;
        T* P7 = workspace + 16 * H2;

        T* T1 = workspace + 17 * H2;
        T* T2 = workspace + 18 * H2;

        T* next_ws = workspace + 19 * H2;
        size_t child_ws_size = get_strassen_workspace_size<T>(H);

        cudaStream_t s[7];
        for (int i = 0; i < 7; i++) {
            CUDA_CHECK(cudaStreamCreate(&s[i]));
        }

        dim3 block_dim(16, 16);
        dim3 grid_dim((H + block_dim.x - 1) / block_dim.x,
                      (H + block_dim.y - 1) / block_dim.y);

        // M1 = (A11 + A22) * (B11 + B22)
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[0]>>>(A11, lda, A22, lda, S1_P1, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[0]>>>(B11, ldb, B22, ldb, S2_P1, H, H);
        _strassen_rec<T>(S1_P1, H, S2_P1, H, P1, H, H, next_ws + 0 * child_ws_size, s[0]);

        // M2 = (A21 + A22) * B11
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[1]>>>(A21, lda, A22, lda, S1_P2, H, H);
        _strassen_rec<T>(S1_P2, H, B11, ldb, P2, H, H, next_ws + 1 * child_ws_size, s[1]);

        // M3 = A11 * (B12 - B22)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[2]>>>(B12, ldb, B22, ldb, S2_P3, H, H);
        _strassen_rec<T>(A11, lda, S2_P3, H, P3, H, H, next_ws + 2 * child_ws_size, s[2]);

        // M4 = A22 * (B21 - B11)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[3]>>>(B21, ldb, B11, ldb, S2_P4, H, H);
        _strassen_rec<T>(A22, lda, S2_P4, H, P4, H, H, next_ws + 3 * child_ws_size, s[3]);

        // M5 = (A11 + A12) * B22
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[4]>>>(A11, lda, A12, lda, S1_P5, H, H);
        _strassen_rec<T>(S1_P5, H, B22, ldb, P5, H, H, next_ws + 4 * child_ws_size, s[4]);

        // M6 = (A21 - A11) * (B11 + B12)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[5]>>>(A21, lda, A11, lda, S1_P6, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[5]>>>(B11, ldb, B12, ldb, S2_P6, H, H);
        _strassen_rec<T>(S1_P6, H, S2_P6, H, P6, H, H, next_ws + 5 * child_ws_size, s[5]);

        // M7 = (A12 - A22) * (B21 + B22)
        sub_square_kernel<T><<<grid_dim, block_dim, 0, s[6]>>>(A12, lda, A22, lda, S1_P7, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, s[6]>>>(B21, ldb, B22, ldb, S2_P7, H, H);
        _strassen_rec<T>(S1_P7, H, S2_P7, H, P7, H, H, next_ws + 6 * child_ws_size, s[6]);

        for (int i = 0; i < 7; i++) {
            CUDA_CHECK(cudaStreamSynchronize(s[i]));
            CUDA_CHECK(cudaStreamDestroy(s[i]));
        }

        // Combine using parent stream
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P1, H, P4, H, T1, H, H);
        sub_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T1, H, P5, H, T2, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T2, H, P7, H, C11, ldc, H);

        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P3, H, P5, H, C12, ldc, H);

        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P2, H, P4, H, C21, ldc, H);

        sub_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(P1, H, P2, H, T1, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T1, H, P3, H, T2, H, H);
        add_square_kernel<T><<<grid_dim, block_dim, 0, stream>>>(T2, H, P6, H, C22, ldc, H);

        return;
    }
}



template <typename T>
__host__ void _gemm_strassen(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
    size_t N = A.shape().first;
    size_t K = A.shape().second;
    size_t M = B.shape().second;

    size_t S = next_pow2(max3(N, K, M));

    if (S <= CUTOFF_SIZE) {
        _gemm_nkm_simple_launcher<T>(A, B, C);
        return;
    }

    T* workspace = nullptr;
    size_t ws_size = get_strassen_workspace_size<T>(S);
    CUDA_CHECK(cudaMalloc(&workspace, ws_size * sizeof(T)));

    if (N == K && K == M && (N == S)) {
        _strassen_rec<T>(A.item(), N, B.item(), K, C.item(), M, S, workspace);
        CUDA_CHECK(cudaFree(workspace));
        return;
    }

    Matrix<T> Ap(S, S, ROW_WISE, CUDA);
    Matrix<T> Bp(S, S, ROW_WISE, CUDA);
    Matrix<T> Cp(S, S, ROW_WISE, CUDA);

    CUDA_CHECK(cudaMemset(Ap.item(), 0, sizeof(T) * S * S));
    CUDA_CHECK(cudaMemset(Bp.item(), 0, sizeof(T) * S * S));
    CUDA_CHECK(cudaMemset(Cp.item(), 0, sizeof(T) * S * S));

    dim3 block_dim(16, 16);
    dim3 gridA((K + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);
    dim3 gridB((M + block_dim.x - 1) / block_dim.x, (K + block_dim.y - 1) / block_dim.y);
    dim3 gridC((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);

    copy_rect_kernel<T><<<gridA, block_dim>>>(A.item(), K, Ap.item(), S, N, K);
    copy_rect_kernel<T><<<gridB, block_dim>>>(B.item(), M, Bp.item(), S, K, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    _strassen_rec<T>(Ap.item(), S, Bp.item(), S, Cp.item(), S, S, workspace);

    copy_rect_kernel<T><<<gridC, block_dim>>>(Cp.item(), S, C.item(), M, N, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(workspace));
}
