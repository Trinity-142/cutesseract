#include "matrix.cuh"

struct MatView {
  fp32 *ptr;
  size_t rows;
  size_t cols;
  size_t ld; // leading dimension

  __host__ __device__ fp32 *row(size_t i) const { return ptr + i * ld; }
};

__host__ __device__ inline MatView subview(MatView m, size_t r0, size_t c0,
                                           size_t r, size_t c) {
  return MatView{m.ptr + r0 * m.ld + c0, r, c, m.ld};
}

template <typename T>
__global__ void matrix_add_kernel(const T *A, const T *B, T *C, size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void matrix_sub_kernel(const T *A, const T *B, T *C, size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    C[idx] = A[idx] - B[idx];
  }
}