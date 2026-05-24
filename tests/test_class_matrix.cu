#include "matrix.cuh"
#include <cassert>

__host__ void test_layout() {
  size_t rows = 5;
  size_t cols = 2;

  auto matrix = Matrix<fp32>(rows, cols, DataLayout::ROW_WISE, DataDevice::CPU);

  assert(matrix.item() != nullptr);
}

__host__ void test_layout_switch() { return; }
