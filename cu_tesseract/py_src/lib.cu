#include <torch/extension.h>

#include "dtypes.cuh"
#include "matrix.cuh"
#include "kernels.cuh"
#include "utils.cuh"


static inline void _check_fp32_cuda_rowwise_matrix(at::Tensor &x) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.stride(1) == 1);
}

void py_gemm_nkm_simple_fp32(
    at::Tensor &A,
    at::Tensor &B,
    at::Tensor &C
) {
    _check_fp32_cuda_rowwise_matrix(A);
    _check_fp32_cuda_rowwise_matrix(B);
    _check_fp32_cuda_rowwise_matrix(C);

    Matrix<fp32> mat_A((fp32 *)A.data_ptr(), A.size(0), A.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> mat_B((fp32 *)B.data_ptr(), B.size(0), B.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> mat_C((fp32 *)C.data_ptr(), C.size(0), C.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);

    _gemm_nkm_simple_launcher<fp32>(mat_A, mat_B, mat_C);
}

template <size_t BS>
void py_gemm_nnn_block_simple_fp32(
    at::Tensor &A,
    at::Tensor &B,
    at::Tensor &C
) {
    _check_fp32_cuda_rowwise_matrix(A);
    _check_fp32_cuda_rowwise_matrix(B);
    _check_fp32_cuda_rowwise_matrix(C);

    TORCH_CHECK((A.size(0) % BS) == 0);

    Matrix<fp32> mat_A((fp32 *)A.data_ptr(), A.size(0), A.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> mat_B((fp32 *)B.data_ptr(), B.size(0), B.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> mat_C((fp32 *)C.data_ptr(), C.size(0), C.size(1), DataLayout::ROW_WISE, DataDevice::CUDA);

    _gemm_nn_block_launcher<fp32>(mat_A, mat_B, mat_C, BS);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_nkm_simple_fp32", &py_gemm_nkm_simple_fp32, "simple gpu gemm for FP32 matrices");
    m.def("gemm_nnn_block_simple_fp32_bs16", &py_gemm_nnn_block_simple_fp32<16>, "simple gpu gemm for FP32 square matrices with shape % 16 == 0");
    m.def("gemm_nnn_block_simple_fp32_bs8", &py_gemm_nnn_block_simple_fp32<8>, "simple gpu gemm for FP32 square matrices with shape % 8 == 0");
    m.def("gemm_nnn_block_simple_fp32_bs4", &py_gemm_nnn_block_simple_fp32<4>, "simple gpu gemm for FP32 square matrices with shape % 4 == 0");
}
