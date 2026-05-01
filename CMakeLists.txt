cmake_minimum_required(VERSION 3.28.3)
project(cutesseract LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xptxas=-v")

include_directories(cu_tesseract/include)

add_executable(cutesseract
        cu_tesseract/src/main.cu)

target_link_libraries(cutesseract curand)
