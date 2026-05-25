
import torch
from py_cutesseract_cuda import (
    gemm_wmma_fp16,
    gemm_wmma_bf16,
    gemm_nkm_simple_fp32,
    gemm_nnn_block_simple_fp32_bs16,
    gemm_nnn_block_simple_fp32_bs8,
    gemm_nnn_block_simple_fp32_bs4,
)

def matmul_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.device == b.device
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    res = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)

    gemm_nkm_simple_fp32(a, b, res)

    return res

def matmul_square_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.device == b.device
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    assert a.shape[0] == a.shape[1] == b.shape[0] == b.shape[1], "Matrices must be square"
    n = a.shape[0]

    res = torch.empty((n, n), dtype=torch.float32, device=a.device)

    if (n % 16) == 0:
        gemm_nnn_block_simple_fp32_bs16(a, b, res)
    elif (n % 8) == 0:
        gemm_nnn_block_simple_fp32_bs8(a, b, res)
    elif (n % 4) == 0:
        gemm_nnn_block_simple_fp32_bs4(a, b, res)
    else:
        print('Warning! Matrix shape is not divisible by at least 4, using slow algo')
        gemm_nkm_simple_fp32(a, b, res)


    return res

def matmul_wmma(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.dim() == 2 and b.dim() == 2
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.dtype == torch.float16 or a.dtype == torch.bfloat16
    assert b.dtype == torch.float16 or a.dtype == torch.bfloat16

    res = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)

    if a.dtype == torch.float16:
        gemm_wmma_fp16(a, b, res)
    else:
        gemm_wmma_bf16(a, b, res)

    return res.to(a.dtype)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.dim() == 2 and b.dim() == 2
    assert a.device == b.device

    if a.dtype == torch.float16 or a.dtype == torch.bfloat16:
        return matmul_wmma(a, b)

    assert a.dtype == torch.float32, "py_cutesseract.matmul supports only FP16, BF16 and FP32"

    if a.shape[0] < 4 or b.shape[1] < 4:
        return matmul_fp32(a, b)

    if a.shape[0] == a.shape[1] == b.shape[0] == b.shape[1]:
        return matmul_square_fp32(a, b)
    else:
        return matmul_fp32(a, b)
