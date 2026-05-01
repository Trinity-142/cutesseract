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
