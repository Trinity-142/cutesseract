FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y cmake g++ make ninja-build
RUN mkdir -p '/root/.config/NVIDIA Corporation' && touch '/root/.config/NVIDIA Corporation/nsys-config.ini'
RUN echo 'HandleInvalidEvents=true\nCuptiUseRawGpuTimestamps=false\n' > '/root/.config/NVIDIA Corporation/nsys-config.ini'
