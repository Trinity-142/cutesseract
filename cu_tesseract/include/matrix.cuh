#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <curand.h>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <string.h>
#include <cuda_runtime.h>

#include "dtypes.cuh"
#include "utils.cuh"


enum class DataLayout {
    ROW_WISE,
    COL_WIZE,
};

enum class DataDevice {
    CPU,
    CUDA,
};

using enum DataLayout;
using enum DataDevice;


template <typename T>
class Matrix {
    T* cpu_ptr;
    T* device_ptr;

    DataLayout layout;
    DataDevice device;

    size_t rows, cols, numel;
public:

    __host__ Matrix(size_t rows, size_t cols, DataLayout layout, DataDevice device): rows(rows), cols(cols), device(device), layout(layout), numel(sizeof(T) * rows * cols) {

        if (device == CUDA) {
            CUDA_CHECK(cudaMalloc(&device_ptr, numel));
            cpu_ptr = nullptr;
        } else {
            cpu_ptr = new T[rows * cols];
            device_ptr = nullptr;
        }

    }

    __host__ ~Matrix() {
        if (device == CUDA) {
            CUDA_CHECK(cudaFree(device_ptr));
        } else {
            delete[] cpu_ptr;
        }
    }

    __host__ Matrix(const Matrix& other): cpu_ptr(nullptr), device_ptr(nullptr), rows(other.rows), cols(other.cols), numel(other.numel), layout(other.layout), device(other.device) {
        if (device == CPU) {
            cpu_ptr = new T[rows * cols];
            memcpy(cpu_ptr, other.cpu_ptr, numel);
        } else {
            CUDA_CHECK(cudaMalloc(&device_ptr, numel));
            cudaMemcpy(device_ptr, other.device_ptr, numel, cudaMemcpyDeviceToDevice);
        }
    }

    __host__ Matrix& operator=(const Matrix& other) {
        rows = other.rows;
        cols = other.cols;
        numel = other.numel;
        layout = other.layout;


        if (device == CPU) {
            delete[] cpu_ptr;
            cpu_ptr = nullptr;
        }
        else {
            CUDA_CHECK(cudaFree(device_ptr));
            device_ptr = nullptr;
        }

        device = other.device;

        if (device == CPU) {
            cpu_ptr = new T[rows * cols];
            memcpy(cpu_ptr, other.cpu_ptr, numel);
        } else {
            CUDA_CHECK(cudaMalloc(&device_ptr, numel));
            cudaMemcpy(device_ptr, other.device_ptr, numel, cudaMemcpyDeviceToDevice);
        }
    }

    __host__ Matrix(Matrix&& other): cpu_ptr(nullptr), device_ptr(nullptr), rows(0), cols(0), numel(0), layout(ROW_WISE), device(CPU) {
        this->swap(other);
    }

    __host__ Matrix& operator=(Matrix&& other) {
        this->swap(other);
        return *this;
    }

    __host__ Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        assert(layout == other.layout);
        assert(device == other.device);

        Matrix result(rows, cols, layout, device);

        if (device == CPU) {
            for (size_t i = 0; i < numel / sizeof(T); ++i) {
                result.cpu_ptr[i] = cpu_ptr[i] + other.cpu_ptr[i];
            }
        } else {
            dim3 block(256);
            dim3 grid((numel / sizeof(T) + block.x - 1) / block.x);
            matrix_add_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr, result.device_ptr, numel / sizeof(T));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        return result;
    }

    __host__ Matrix operator-(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        assert(layout == other.layout);
        assert(device == other.device);

        Matrix result(rows, cols, layout, device);

        if (device == CPU) {
            for (size_t i = 0; i < numel / sizeof(T); ++i) {
                result.cpu_ptr[i] = cpu_ptr[i] - other.cpu_ptr[i];
            }
        } else {
            dim3 block(256);
            dim3 grid((numel / sizeof(T) + block.x - 1) / block.x);
            matrix_sub_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr, result.device_ptr, numel / sizeof(T));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        return result;
    }

    __host__ Matrix& operator+=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        assert(layout == other.layout);
        assert(device == other.device);

        if (device == CPU) {
            for (size_t i = 0; i < numel / sizeof(T); ++i) {
                cpu_ptr[i] += other.cpu_ptr[i];
            }
        } else {
            dim3 block(256);
            dim3 grid((numel / sizeof(T) + block.x - 1) / block.x);
            matrix_add_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr, device_ptr, numel / sizeof(T));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        return *this;
    }

    __host__ Matrix& operator-=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        assert(layout == other.layout);
        assert(device == other.device);

        if (device == CPU) {
            for (size_t i = 0; i < numel / sizeof(T); ++i) {
                cpu_ptr[i] -= other.cpu_ptr[i];
            }
        } else {
            dim3 block(256);
            dim3 grid((numel / sizeof(T) + block.x - 1) / block.x);
            matrix_sub_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr, device_ptr, numel / sizeof(T));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        return *this;
    }

    __host__ void swap(Matrix& other) {
        std::swap(cpu_ptr, other.cpu_ptr);
        std::swap(device_ptr, other.device_ptr);
        std::swap(layout, other.layout);
        std::swap(device, other.device);
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        std::swap(numel, other.numel);
    }

    __host__ void fill_random(unsigned long long seed = 812ULL) {
        if (device == CUDA) {
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, seed);
            assert (sizeof(T) == sizeof(fp32))
            curandGenerateUniform(gen, device_ptr, rows * cols);
            // else
            //     curandGenerateUniformDouble(gen, device_ptr, rows * cols);
            curandDestroyGenerator(gen);
        } else {
            throw std::runtime_error("Random init not implemented for cpu");
        }
    }

    __host__ void ones() {
        assert(device == CPU && layout == ROW_WISE);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                cpu_ptr[i * cols + j] = (T)1.0;
            }
        }
    }

    __host__ void cpu() {
        if (device == CPU) return;

        device = CPU;

        cpu_ptr = new T[cols * rows];
        CUDA_CHECK(cudaMemcpy(cpu_ptr, device_ptr, numel, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(device_ptr));

        device_ptr = nullptr;
    }

    __host__ void cuda() {
        if (device == CUDA) return;

        device = CUDA;

        CUDA_CHECK(cudaMalloc(&device_ptr, numel));
        CUDA_CHECK(cudaMemcpy(device_ptr, cpu_ptr, numel, cudaMemcpyHostToDevice));

        delete[] cpu_ptr;
        cpu_ptr = nullptr;
    }

    __host__ T* item() const {
        if (device == CPU) {
            return cpu_ptr;
        } else {
            return device_ptr;
        }
    }

    __host__ void to_layout(DataLayout new_layout) {
        if (layout == new_layout) return;

        T* new_buffer = new T[rows * cols];

        if (device == CUDA) {
            delete[] new_buffer;
            throw std::runtime_error(".to_layout not implemented for CUDA. consider using .cpu()");
        } else {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    if (new_layout == ROW_WISE) {
                        new_buffer[i + j * cols] = cpu_ptr[i * rows + j];
                    } else {
                        new_buffer[i * rows + j] = cpu_ptr[i + j * cols];
                    }
                }
            }
        }

        delete[] cpu_ptr;
        cpu_ptr = new_buffer;

        layout = new_layout;
    }

    __host__ friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        if (matrix.device == CUDA) {
            throw std::runtime_error("data must be on cpu for printing. consider calling .cpu()");
        }

        for (size_t i = 0; i < matrix.rows; i++) {
            os << "[";
            for (size_t j = 0; j < matrix.cols; j++) {
                os << matrix.get(i, j);
                if (j != matrix.cols - 1) os << ", ";
            }
            os << "]\n";
        }

        return os;
    }

    __host__ std::pair<size_t, size_t> shape() const {
        return {rows, cols};
    }

    __host__ DataLayout get_layout() const {
        return layout;
    }

    __host__ T get(size_t i, size_t j) const {
        /* row-wise getter */

        if (i >= rows || j >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        if (device == CUDA) {
            throw std::runtime_error("data must be on cpu to get value. consider calling .cpu()");
        }

        if (layout == ROW_WISE) {
            return cpu_ptr[i * cols + j];
        } else {
            return cpu_ptr[i + j * rows];
        }
    }
};

#endif
