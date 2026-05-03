
import torch
from py_cutesseract_cuda import (
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
