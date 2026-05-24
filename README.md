# cuTesseract

### usage

```shell
docker build --network host -t cutesseract-env docker/
docker run --rm -it --gpus all --cap-add=SYS_ADMIN --network host --ipc host -v $(pwd):/workspace -w /workspace cutesseract-env /bin/bash
# mkdir build && cd build
# cmake -G Ninja ..
# ninja
# ./cutesseract
COMPUTE_CAP=86 pip install . --no-build-isolation
```

### Tests
```
Blockwise GPU multiplication duration: ~51.7303ms
TFLOPS: 1.12
Elementwise GPU multiplication duration: ~68.03ms
TFLOPS: 0.85
Warp Matrix Multiply-Accumulate GPU multiplication duration: ~4.11489ms
TFLOPS: 14.09
```

```python
import torch
from py_cutesseract import matmul_fp32, matmul_square_fp32

def main():
    a = torch.rand((4, 4), device='cuda:0')
    b = torch.rand((4, 4), device='cuda:0')

    print(matmul_square_fp32(a, b))
    print(a @ b)

    a = torch.rand((12, 42), device='cuda:0')
    b = torch.rand((42, 4), device='cuda:0')

    print(matmul_fp32(a, b))
    print(a @ b)

if __name__ == '__main__':
    main()

```
