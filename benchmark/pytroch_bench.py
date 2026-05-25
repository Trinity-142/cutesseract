
import gc
import time
import torch
from typing import Callable
# from py_cutesseract import matmul_fp32, matmul_square_fp32
from py_cutesseract_cuda import gemm_nkm_simple_fp32, gemm_nnn_block_simple_fp32_bs16, gemm_wmma_fp16

def bench_torch(a, b):
    assert a.dim() == b.dim() == 3
    assert a.shape[0] == b.shape[0]

    num_samples = a.shape[0]
    res = torch.zeros(num_samples, a.shape[1], b.shape[2], device=a.device, dtype=a.dtype)
    for _ in range(3):
        torch.matmul(a[0], b[0], out=res[0])
    torch.cuda.synchronize()

    cnt = 0.0
    for i in range(num_samples):
        torch.cuda.synchronize()
        now = time.time()
        torch.matmul(a[i], b[i], out=res[i])
        torch.cuda.synchronize()
        cnt += time.time() - now

    print(f'Time spent avg {cnt / num_samples * 1000:.6f} mcs')

def bench_cutesseract(method: Callable, a, b, out_dtype = None):
    assert a.dim() == b.dim() == 3
    assert a.shape[0] == b.shape[0]

    num_samples = a.shape[0]
    res = torch.zeros(num_samples, a.shape[1], b.shape[2], device=a.device, dtype=(a.dtype if out_dtype is None else out_dtype))

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
    rect_samples = generate_samples(1, (2048, 8192, 2048), device='cuda:0', dtype=torch.float32)
    rect_samples_cpu = generate_samples(1, (2048, 8192, 2048), device='cpu', dtype=torch.float32)
    square_samples = generate_samples(1, (128, 128, 128), device='cuda:0', dtype=torch.float32)

    print('WARMUP: ', end='')
    bench_cutesseract(gemm_nnn_block_simple_fp32_bs16, *square_samples)

    print('torch rect: ', end='')
    bench_torch(*rect_samples)

    print('torch rect cpu: ', end='')
    bench_torch(*rect_samples_cpu)

    print('torch square: ', end='')
    bench_torch(*square_samples)

    print('cutesseract rect: ', end='')
    bench_cutesseract(gemm_nkm_simple_fp32, *rect_samples)

    print('cutesseract square: ', end='')
    bench_cutesseract(gemm_nnn_block_simple_fp32_bs16, *square_samples)

    del rect_samples, square_samples
    gc.collect()

    rect_samples = generate_samples(1, (2048, 8192, 2048), device='cuda:0', dtype=torch.float16)

    print('torch rect fp16: ', end='')
    bench_torch(*rect_samples)

    print('cutesseract rect wmma fp16: ', end='')
    bench_cutesseract(gemm_wmma_fp16, *rect_samples, out_dtype=torch.float32)

    del rect_samples
    gc.collect()

    rect_samples = generate_samples(1, (2048, 8192, 2048), device='cuda:0', dtype=torch.bfloat16)

    print('torch rect bf16: ', end='')
    bench_torch(*rect_samples)

    print('cutesseract rect wmma bf16: ', end='')
    bench_cutesseract(gemm_wmma_fp16, *rect_samples, out_dtype=torch.float32)

if __name__ == '__main__':
    main()
