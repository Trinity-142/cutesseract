#include "matrix.cuh"
#include "kernels.cuh"
#include "strassen_kernel.cuh"
#include "utils.cuh"

#include <iostream>
#include <chrono>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

constexpr size_t n = 3072, k = 3072, m = 3072;
// constexpr size_t N = 256;


void verify_cpu(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
  A.cpu();
  B.cpu();
  C.cpu();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      fp32 sum = 0.0;
      for (size_t r = 0; r < k; r++) {
        sum += A.get(i, r) * B.get(r, j);
      }

      if (std::abs(sum - C.get(i, j)) >= 1e-4) {
        cout << sum << ' ' << C.get(i, j) << " (" << i << ", " << j << ")\n";
        throw std::runtime_error("verification failed");
      }
    }
  }
}

std::chrono::duration<double, std::milli>
test_blockwise(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
  auto start_time = std::chrono::high_resolution_clock::now();

  _gemm_nn_block_launcher<fp32>(A, B, C, 16);

  std::chrono::duration<double, std::milli> res = std::chrono::high_resolution_clock::now() - start_time;

  // verify_cpu(A, B, C);
  return res;
}

std::chrono::duration<double, std::milli>
test_elementwise(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
  auto start_time = std::chrono::high_resolution_clock::now();

  _gemm_nkm_simple_launcher<fp32>(A, B, C);

  std::chrono::duration<double, std::milli> res = std::chrono::high_resolution_clock::now() - start_time;

  // verify_cpu(A, B, C);
  return res;
}

std::chrono::duration<double, std::milli> test_wmma(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    A.cuda();
    B.cuda();
    C.cuda();
    Matrix<half> A_fp16(n, k, ROW_WISE, CUDA);
    Matrix<half> B_fp16(k, m, ROW_WISE, CUDA);

    size_t threads = 256;
    castFp32ToFp16<<<(n * k + threads - 1) / threads, threads>>>(A.item(), A_fp16.item(), n * k);
    castFp32ToFp16<<<(k * m + threads - 1) / threads, threads>>>(B.item(), B_fp16.item(), k * m);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start_time = std::chrono::high_resolution_clock::now();

    _gemm_nn_wmma_launcher<n, k, m>(A_fp16, B_fp16, C);

    std::chrono::duration<double, std::milli> res = std::chrono::high_resolution_clock::now() - start_time;

    // verify_cpu(A, B, C);
    return res;
}

std::chrono::duration<double, std::milli>
test_strassen(Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
  auto start_time = std::chrono::high_resolution_clock::now();

  _gemm_strassen_launcher<fp32>(A, B, C);

  return std::chrono::high_resolution_clock::now() - start_time;

  verify_cpu(A, B, C);
}

signed main() {

  size_t num_tries = 16;

  vector<Matrix<fp32> *> input_matrices_a;
  vector<Matrix<fp32> *> input_matrices_b;
  vector<Matrix<fp32> *> input_matrices_c;

  Matrix<fp32> *A, *B, *C;

  for (size_t i = 0; i < num_tries + 1; i++) {
    A = new Matrix<fp32>((size_t)n, (size_t)k, ROW_WISE, CUDA);
    B = new Matrix<fp32>((size_t)k, (size_t)m, ROW_WISE, CUDA);
    C = new Matrix<fp32>((size_t)n, (size_t)m, ROW_WISE, CUDA);

    A->fill_random((unsigned long long)(i + 993));
    B->fill_random((unsigned long long)(i + 993));

    input_matrices_a.push_back(A);
    input_matrices_b.push_back(B);
    input_matrices_c.push_back(C);
  }

    std::chrono::duration<double, std::milli> avg_block = std::chrono::duration<double, std::milli>::zero();
    std::chrono::duration<double, std::milli> avg_element = std::chrono::duration<double, std::milli>::zero();
    std::chrono::duration<double, std::milli> avg_wmma = std::chrono::duration<double, std::milli>::zero();

    test_blockwise(*input_matrices_a[num_tries], *input_matrices_b[num_tries], *input_matrices_c[num_tries]);
    test_elementwise(*input_matrices_a[num_tries], *input_matrices_b[num_tries], *input_matrices_c[num_tries]);
    test_wmma(*input_matrices_a[num_tries], *input_matrices_b[num_tries], *input_matrices_c[num_tries]);

    for (size_t i = 0; i < num_tries; i++) {
        avg_block += test_blockwise(*input_matrices_a[i], *input_matrices_b[i], *input_matrices_c[i]);
    }

    cout << "Blockwise GPU multiplication duration: ~" << avg_block / (num_tries) << "\n";

    for (size_t i = 0; i < num_tries; i++) {
        avg_element += test_elementwise(*input_matrices_a[i], *input_matrices_b[i], *input_matrices_c[i]);
    }

  cout << "Elementwise GPU multiplication duration: ~" << avg_element / (num_tries) << "\n";

  for (size_t i = 0; i < num_tries; i++) {
    avg_element += test_strassen(*input_matrices_a[i], *input_matrices_b[i], *input_matrices_c[i]);
  }

  cout << "Strassen GPU multiplication duration: ~" << avg_element / (num_tries) << "\n";

  return 0;
  for (size_t i = 0; i < num_tries; i++) {
      avg_wmma += test_wmma(*input_matrices_a[i], *input_matrices_b[i], *input_matrices_c[i]);
  }

  cout << "Warp Matrix Multiply-Accumulate GPU multiplication duration: ~" << avg_wmma / (num_tries) << "\n";

  return 0;
}

// nsys profile --gpu-metrics-devices=all --cpuctxsw=process-tree
// --sample=process-tree -o test_profile ./cutesseract
