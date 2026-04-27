# cuTesseract

### usage

```Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    gdb \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
```

```shell
docker build -t cutesseract-env .
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace cutesseract-env /bin/bash
mkdir build && cd build
cmake -G Ninja ..
ninja
./cutesseract
```

### Tests
```
Blockwise GPU multiplication duration: ~51.3392ms
TFLOPS: 1.13
Elementwise GPU multiplication duration: ~68.5498ms
TFLOPS: 0.85
Warp Matrix Multiply-Accumulate GPU multiplication duration: ~8.29288ms
TFLOPS: 7.03
```
