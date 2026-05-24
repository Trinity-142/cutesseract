
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "matrix.cuh"
#include "strassen_kernel.cuh"
#include "test_class_matrix.cu"

using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::map;
using std::string;
using std::vector;

#ifndef RUNS_NUM
#define RUNS_NUM 4
#endif

enum class FillType { RANDOM, ONES, ZEROS };

typedef function<void(Matrix<fp32> &, Matrix<fp32> &, Matrix<fp32> &)>
    KernelFunc;

template <typename T> T calculate_max_diff(Matrix<T> &A, Matrix<T> &B) {
  assert(A.device == CPU && B.device == CPU);
  std::pair<size_t, size_t> shapeA = A.shape();
  std::pair<size_t, size_t> shapeB = B.shape();
  assert(shapeA == shapeB);
  T max_diff = 0.0;
  for (size_t i = 0; i < shapeA.first; i++) {
    for (size_t j = 0; j < shapeA.second; j++) {
      T diff = std::abs(A.get(i, j) - B.get(i, j));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  return max_diff;
}

template <typename T> Matrix<T> mmul_cpu(Matrix<T> &A, Matrix<T> &B) {
  assert(A.device == CPU && B.device == CPU);
  std::pair<size_t, size_t> shapeA = A.shape();
  std::pair<size_t, size_t> shapeB = B.shape();

  assert(shapeA.second == shapeB.first);
  Matrix<T> C(shapeA.first, shapeB.second, ROW_WISE, CPU);
  for (size_t i = 0; i < shapeA.first; i++) {
    for (size_t j = 0; j < shapeB.second; j++) {
      double sum = 0.0;
      for (size_t r = 0; r < shapeA.second; r++) {
        sum += (double)A.get(i, r) * (double)B.get(r, j);
      }
      C.set(i, j, (T)sum);
    }
  }
  return C;
}

template <typename T>
void print_heatmap(Matrix<T> &GPU_C, Matrix<T> &CPU_C, T precision) {
  std::pair<size_t, size_t> shapeGPU = GPU_C.shape();
  std::pair<size_t, size_t> shapeCPU = CPU_C.shape();
  assert(shapeGPU == shapeCPU);
  assert(GPU_C.device == CPU && CPU_C.device == CPU);
  size_t rows = shapeGPU.first;
  size_t cols = shapeGPU.second;
  size_t grid_r = std::min(rows, (size_t)32);
  size_t grid_c = std::min(cols, (size_t)32);
  size_t step_r = (rows + grid_r - 1) / grid_r;
  size_t step_c = (cols + grid_c - 1) / grid_c;

  cout << "\nError Heatmap (" << grid_r << "x" << grid_c
       << " sampling):" << endl;
  for (size_t i = 0; i < grid_r; i++) {
    for (size_t j = 0; j < grid_c; j++) {
      bool has_error = false;
      for (size_t bi = i * step_r; bi < std::min((i + 1) * step_r, rows); bi++) {
        for (size_t bj = j * step_c; bj < std::min((j + 1) * step_c, cols); bj++) {
          if (std::abs(GPU_C.get(bi, bj) - CPU_C.get(bi, bj)) > precision) {
            has_error = true;
            break;
          }
        }
        if (has_error)
          break;
      }
      cout << (has_error ? "X" : ".");
    }
    cout << endl;
  }
}

void verify_result(Matrix<fp32> &GPU_C, Matrix<fp32> &CPU_C,
                   fp32 precision = 1e-3) {
  assert(GPU_C.device == CPU && CPU_C.device == CPU);
  fp32 max_diff = calculate_max_diff(GPU_C, CPU_C);
  if (max_diff > precision) {
    cout << "[FAILED] Max difference: " << std::scientific << max_diff << endl;
    print_heatmap(GPU_C, CPU_C, precision);
  } else {
    cout << "[PASSED] Max difference: " << std::scientific << max_diff << endl;
  }
}

void run_test(KernelFunc kernel, size_t N, size_t K, size_t M, FillType fill,
              int runs = RUNS_NUM) {
  for (int i = 0; i < runs; i++) {
    Matrix<fp32> A(N, K, ROW_WISE, CUDA);
    Matrix<fp32> B(K, M, ROW_WISE, CUDA);
    Matrix<fp32> G(N, M, ROW_WISE, CUDA);

    if (fill == FillType::RANDOM) {
      A.fill_random((unsigned long long)i);
      B.fill_random((unsigned long long)i + 1337);
    } else if (fill == FillType::ONES) {
      A.ones();
      B.ones();
    } else if (fill == FillType::ZEROS) {
      A.zeros();
      B.zeros();
    }

    kernel(A, B, G);

    A.cpu();
    B.cpu();
    G.cpu();
    Matrix<fp32> C = mmul_cpu(A, B);
    verify_result(G, C);
  }
}

void run_benchmark(map<string, KernelFunc> &registry, size_t size = 1024,
                   int trials = 10) {
  cout << "\n--- Benchmarking Kernels (Size: " << size << "x" << size
       << ", Trials: " << trials << ") ---" << endl;

  map<string, double> accumulated_times;

  for (int t = 0; t < trials; t++) {
    Matrix<fp32> A(size, size, ROW_WISE, CUDA);
    Matrix<fp32> B(size, size, ROW_WISE, CUDA);
    A.fill_random((unsigned long long)t);
    B.fill_random((unsigned long long)t + 1337);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto const &[name, kernel] : registry) {
      Matrix<fp32> C(size, size, ROW_WISE, CUDA);
      auto start = std::chrono::high_resolution_clock::now();
      kernel(A, B, C);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end - start;
      accumulated_times[name] += duration.count();
    }
  }

  for (auto const &[name, kernel] : registry) {
    cout << std::left << std::setw(12) << name << ": " << std::fixed
         << std::setprecision(3) << accumulated_times[name] / trials << " ms"
         << endl;
  }
}

void iterative_stress_test(KernelFunc kernel) {
  for (size_t size = 16; size <= 1024; size *= 2) {
    cout << "\n--- Size: " << size << "x" << size << " ---" << endl;
    run_test(kernel, size, size, size, FillType::RANDOM, 1);
  }
}

void menu() {
  map<string, KernelFunc> kernel_registry;
  kernel_registry["Simple"] = _gemm_nkm_simple_launcher<fp32>;
  kernel_registry["Blocked"] = [](Matrix<fp32> &A, Matrix<fp32> &B,
                                  Matrix<fp32> &C) {
    _gemm_nn_block_launcher<fp32>(A, B, C);
  };
  kernel_registry["Strassen"] = _gemm_strassen_launcher<fp32>;
  kernel_registry["WMMA"] = [](Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    size_t N = A.shape().first;
    size_t K = A.shape().second;
    size_t M = B.shape().second;

    Matrix<fp16> A_fp16(N, K, ROW_WISE, CUDA);
    Matrix<fp16> B_fp16(K, M, ROW_WISE, CUDA);

    size_t threads = 256;
    castFp32ToFp16<<<(N * K + threads - 1) / threads, threads>>>(A.item(), A_fp16.item(), N * K);
    castFp32ToFp16<<<(K * M + threads - 1) / threads, threads>>>(B.item(), B_fp16.item(), K * M);
    CUDA_CHECK(cudaDeviceSynchronize());
    _gemm_nkm_wmma_launcher(A_fp16, B_fp16, C);
  };

  while (true) {
    cout << "\n=== CuTesseract Test CLI ===" << endl;
    cout << "1. Run Performance Benchmark (1024x1024)" << endl;
    cout << "2. Standard Kernel Verification (512x512)" << endl;
    cout << "3. Iterative Stress Test (16->1024)" << endl;
    cout << "4. Exit" << endl;
    cout << "Choice: ";

    int choice;
    if (!(cin >> choice))
      break;

    if (choice == 1) {
      run_benchmark(kernel_registry);
    } else if (choice == 2 || choice == 3) {
      cout << "\nSelect Kernel:" << endl;
      int idx = 1;
      vector<string> names;
      for (auto const &[name, func] : kernel_registry) {
        cout << idx++ << ". " << name << endl;
        names.push_back(name);
      }
      int k_choice;
      cin >> k_choice;
      if (k_choice < 1 || k_choice > names.size())
        continue;
      KernelFunc kernel = kernel_registry[names[k_choice - 1]];

      cout << "Select Fill:\n1. Random\n2. Ones\n3. Zeros\nChoice: ";
      int f_choice;
      cin >> f_choice;
      FillType fill = (f_choice == 2 ? FillType::ONES
                                     : (f_choice == 3 ? FillType::ZEROS
                                                      : FillType::RANDOM));

      if (choice == 2)
        run_test(kernel, 512, 512, 512, fill);
      else
        iterative_stress_test(kernel);
    } else if (choice == 4)
      break;
  }
}

int main() {
  menu();
  return 0;
}
