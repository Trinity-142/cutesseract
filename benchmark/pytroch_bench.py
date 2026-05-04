
import time
import torch
from typing import Callable
# from py_cutesseract import matmul_fp32, matmul_square_fp32
from py_cutesseract_cuda import gemm_nkm_simple_fp32, gemm_nnn_block_simple_fp32_bs16

def bench_torch(a, b):
    assert a.dim() == b.dim() == 3
    assert a.shape[0] == b.shape[0]

    num_samples = a.shape[0]
    res = torch.zeros(num_samples, a.shape[1], b.shape[2], device=a.device, dtype=a.dtype)

    cnt = 0.0
    for i in range(num_samples):
        now = time.time()
        torch.matmul(a[i], b[i], out=res[i])
        cnt += time.time() - now

    print(f'Time spent avg {cnt / num_samples * 1000:.6f} mcs')

def bench_cutesseract(method: Callable, a, b):
    assert a.dim() == b.dim() == 3
    assert a.shape[0] == b.shape[0]

    num_samples = a.shape[0]
    res = torch.zeros(num_samples, a.shape[1], b.shape[2], device=a.device, dtype=a.dtype)

    cnt = 0.0
    
    for i in range(num_samples):
        now = time.time()
        method(a[i], b[i], res[i])
        cnt += time.time() - now

    print(f'Time spent avg {cnt / num_samples * 1000:.6f} mcs')

def generate_samples(
    num_sampels,
    sizes: tuple[int, int, int],
    device=torch.device('cuda:0'),
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.rand(num_sampels, sizes[0], sizes[1], device=device, dtype=dtype).contiguous()
    b = torch.rand(num_sampels, sizes[1], sizes[2], device=device, dtype=dtype).contiguous()

    return a, b



def main():
    rect_samples = generate_samples(10, (1024, 2048, 1024), device='cuda:0', dtype=torch.float32)
    square_samples = generate_samples(10, (4096, 4096, 4096), device='cuda:0', dtype=torch.float32)

    print('torch rect: ', end='')
    bench_torch(*rect_samples)

    print('torch square: ', end='')
    bench_torch(*square_samples)

    print('cutesseract rect: ', end='')
    bench_cutesseract(gemm_nkm_simple_fp32, *rect_samples)

    print('cutesseract square: ', end='')
    bench_cutesseract(gemm_nnn_block_simple_fp32_bs16, *square_samples)

if __name__ == '__main__':
    main()
