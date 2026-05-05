#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string.h>
#include <utility>

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

template <typename T> class Matrix {
  T *cpu_ptr;
  T *device_ptr;

  DataLayout layout;
  DataDevice device;

  size_t rows, cols, num_elements;

public:
  __host__ Matrix(size_t rows, size_t cols, DataLayout layout,
                  DataDevice device)
      : rows(rows), cols(cols), device(device), layout(layout),
        num_elements(sizeof(T) * rows * cols) {

    if (device == CUDA) {
      CUDA_CHECK(cudaMalloc(&device_ptr, num_elements));
      cpu_ptr = nullptr;
    } else {
      cpu_ptr = new T[rows * cols];
      device_ptr = nullptr;
    }
  }

  __host__ ~Matrix() {
    if (device == CUDA) {
      if (device_ptr)
        CUDA_CHECK(cudaFree(device_ptr));
    } else {
      if (cpu_ptr)
        delete[] cpu_ptr;
    }
  }

  __host__ Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols), num_elements(other.num_elements),
        layout(other.layout), device(other.device), cpu_ptr(nullptr),
        device_ptr(nullptr) {
    if (device == CPU) {
      cpu_ptr = new T[rows * cols];
      memcpy(cpu_ptr, other.cpu_ptr, num_elements);
    } else {
      CUDA_CHECK(cudaMalloc(&device_ptr, num_elements));
      CUDA_CHECK(cudaMemcpy(device_ptr, other.device_ptr, num_elements,
                            cudaMemcpyDeviceToDevice));
    }
  }

  __host__ Matrix &operator=(const Matrix &other) {
    if (this == &other)
      return *this;

    if (device == CPU) {
      if (cpu_ptr)
        delete[] cpu_ptr;
    } else {
      if (device_ptr)
        CUDA_CHECK(cudaFree(device_ptr));
    }

    rows = other.rows;
    cols = other.cols;
    num_elements = other.num_elements;
    layout = other.layout;
    device = other.device;

    if (device == CPU) {
      cpu_ptr = new T[rows * cols];
      memcpy(cpu_ptr, other.cpu_ptr, num_elements);
      device_ptr = nullptr;
    } else {
      CUDA_CHECK(cudaMalloc(&device_ptr, num_elements));
      CUDA_CHECK(cudaMemcpy(device_ptr, other.device_ptr, num_elements,
                            cudaMemcpyDeviceToDevice));
      cpu_ptr = nullptr;
    }

    return *this;
  }

  __host__ Matrix(Matrix &&other)
      : cpu_ptr(nullptr), device_ptr(nullptr), rows(0), cols(0),
        num_elements(0), layout(ROW_WISE), device(CPU) {
    this->swap(other);
  }

  __host__ Matrix &operator=(Matrix &&other) {
    if (this == &other)
      return *this;
    this->swap(other);
    return *this;
  }

  __host__ Matrix operator+(const Matrix &other) const {
    assert(rows == other.rows && cols == other.cols);
    assert(layout == other.layout);
    assert(device == other.device);

    Matrix result(rows, cols, layout, device);

    if (device == CPU) {
      for (size_t i = 0; i < rows * cols; ++i) {
        result.cpu_ptr[i] = cpu_ptr[i] + other.cpu_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_add_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr,
                                            result.device_ptr,
                                            num_elements / sizeof(T));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return result;
  }

  __host__ Matrix operator-(const Matrix &other) const {
    assert(rows == other.rows && cols == other.cols);
    assert(layout == other.layout);
    assert(device == other.device);

    Matrix result(rows, cols, layout, device);

    if (device == CPU) {
      for (size_t i = 0; i < rows * cols; ++i) {
        result.cpu_ptr[i] = cpu_ptr[i] - other.cpu_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_sub_kernel<T><<<grid, block>>>(device_ptr, other.device_ptr,
                                            result.device_ptr,
                                            num_elements / sizeof(T));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return result;
  }

  __host__ Matrix &operator+=(const Matrix &other) {
    assert(rows == other.rows && cols == other.cols);
    assert(layout == other.layout);
    assert(device == other.device);

    if (device == CPU) {
      for (size_t i = 0; i < rows * cols; ++i) {
        cpu_ptr[i] += other.cpu_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_add_kernel<T><<<grid, block>>>(
          device_ptr, other.device_ptr, device_ptr, num_elements / sizeof(T));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return *this;
  }

  __host__ Matrix &operator-=(const Matrix &other) {
    assert(rows == other.rows && cols == other.cols);
    assert(layout == other.layout);
    assert(device == other.device);

    if (device == CPU) {
      for (size_t i = 0; i < rows * cols; ++i) {
        cpu_ptr[i] -= other.cpu_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_sub_kernel<T><<<grid, block>>>(
          device_ptr, other.device_ptr, device_ptr, num_elements / sizeof(T));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return *this;
  }

  __host__ void swap(Matrix &other) {
    std::swap(cpu_ptr, other.cpu_ptr);
    std::swap(device_ptr, other.device_ptr);
    std::swap(layout, other.layout);
    std::swap(device, other.device);
    std::swap(rows, other.rows);
    std::swap(cols, other.cols);
    std::swap(num_elements, other.num_elements);
  }

  __host__ void fill_random(unsigned long long seed = 812ULL) {
    if (device == CUDA) {
      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(gen, seed);
      assert(sizeof(T) == sizeof(fp32));
      curandGenerateUniform(gen, device_ptr, rows * cols);
      // else
      //     curandGenerateUniformDouble(gen, device_ptr, rows * cols);
      curandDestroyGenerator(gen);
    } else {
      std::mt19937 gen(seed);
      std::uniform_real_distribution<T> dis(0.0, 1.0);
      for (size_t i = 0; i < rows * cols; i++)
        cpu_ptr[i] = dis(gen);
    }
  }

  __host__ void fill_const(T val) {
    if (device == CUDA) {
      T *h_ptr = new T[rows * cols];
      for (size_t i = 0; i < rows * cols; i++)
        h_ptr[i] = val;
      CUDA_CHECK(
          cudaMemcpy(device_ptr, h_ptr, num_elements, cudaMemcpyHostToDevice));
      delete[] h_ptr;
    } else {
      for (size_t i = 0; i < rows * cols; i++)
        cpu_ptr[i] = val;
    }
  }

  __host__ void ones() { fill_const((T)1.0); }

  __host__ void zeros() {
    if (device == CUDA) {
      CUDA_CHECK(cudaMemset(device_ptr, 0, num_elements));
    } else {
      memset(cpu_ptr, 0, num_elements);
    }
  }

  __host__ void cpu() {
    if (device == CPU)
      return;

    device = CPU;

    cpu_ptr = new T[cols * rows];
    CUDA_CHECK(
        cudaMemcpy(cpu_ptr, device_ptr, num_elements, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_ptr));

    device_ptr = nullptr;
  }

  __host__ void cuda() {
    if (device == CUDA)
      return;

    device = CUDA;

    CUDA_CHECK(cudaMalloc(&device_ptr, num_elements));
    CUDA_CHECK(
        cudaMemcpy(device_ptr, cpu_ptr, num_elements, cudaMemcpyHostToDevice));

    delete[] cpu_ptr;
    cpu_ptr = nullptr;
  }

  __host__ T *item() const {
    if (device == CPU) {
      return cpu_ptr;
    } else {
      return device_ptr;
    }
  }

  __host__ void to_layout(DataLayout new_layout) {
    if (layout == new_layout)
      return;

    if (device == CUDA) {
      throw std::runtime_error(
          ".to_layout not implemented for CUDA. consider using .cpu()");
    }

    T *new_buffer = new T[rows * cols];
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        if (new_layout == ROW_WISE) {
          // Current is COL_WISE: i + j * rows
          new_buffer[i * cols + j] = cpu_ptr[i + j * rows];
        } else {
          // Current is ROW_WISE: i * cols + j
          new_buffer[i + j * rows] = cpu_ptr[i * cols + j];
        }
      }
    }

    delete[] cpu_ptr;
    cpu_ptr = new_buffer;
    layout = new_layout;
  }

  __host__ friend std::ostream &operator<<(std::ostream &os,
                                           const Matrix &matrix) {
    if (matrix.device == CUDA) {
      throw std::runtime_error(
          "data must be on cpu for printing. consider calling .cpu()");
    }

    for (size_t i = 0; i < matrix.rows; i++) {
      os << "[";
      for (size_t j = 0; j < matrix.cols; j++) {
        os << matrix.get(i, j);
        if (j != matrix.cols - 1)
          os << ", ";
      }
      os << "]\n";
    }

    return os;
  }

  __host__ std::pair<size_t, size_t> shape() const { return {rows, cols}; }

  __host__ DataLayout get_layout() const { return layout; }

  __host__ T get(size_t i, size_t j) const {
    /* row-wise getter */

    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    if (device == CUDA) {
      throw std::runtime_error(
          "data must be on cpu to get value. consider calling .cpu()");
    }

    if (layout == ROW_WISE) {
      return cpu_ptr[i * cols + j];
    } else {
      return cpu_ptr[i + j * rows];
    }
  }

  __host__ void set(size_t i, size_t j, T val) {
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    if (device == CUDA) {
      throw std::runtime_error(
          "data must be on cpu to set value. consider calling .cpu()");
    }

    if (layout == ROW_WISE) {
      cpu_ptr[i * cols + j] = val;
    } else {
      cpu_ptr[i + j * rows] = val;
    }
  }
};

#endif
