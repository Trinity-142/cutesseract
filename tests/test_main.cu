
#include <iostream>
#include <cmath>

#include "matrix.cuh"

#include "test_class_matrix.cu"

#ifndef RUNS_NUM
#define RUNS_NUM 1024
#endif

using std::cout;
using std::endl;
using std::vector;

template <typename T>
T froebenius(Matrix<T> A, Matrix<T> B) {
    A.cpu();
    B.cpu();
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    T sumA = 0.0;
    T sumB = 0.0;
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            sumA += A.get(i, j) * A.get(i, j);
            sumB += B.get(i, j) * B.get(i, j);
        }
    }
    return std::abs(sumA - sumB); 
}

std::vector<std::pair<unsigned, unsigned> > diffs(Matrix<T> A, Matrix<T> B) {
    A.cpu();
    B.cpu();
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    vector<std::pair<unsigned, unsigned> > diffs;
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            if (std::abs(A.get(i, j) - B.get(i, j)) > 1e-3) {
                diffs.push_back({i, j});
            }
        }
    }
    return diffs;
}

Matrix<T> mmul_cpu(Matrix<T> A, Matrix<T> B) {
    assert(A.cols() == B.rows());
    Matrix<T> C(A.rows(), B.cols(), ROW_WISE, CPU);
    for(size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < B.cols(); j++) {
            T sum = 0.0;
            for (size_t r = 0; r < A.cols(); r++) {
                sum += A.get(i, r) * B.get(r, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}


signed test(unsigned rows1, unsigned cols1, unsigned rows2, unsigned cols2, (void) (*mul_function)(Matrix<T>, Matrix<T>, Matrix<T>)) {
    assert(rows1 == cols2 && "геи\n");
    unsigned runs = RUNS_NUM;
        for (unsigned i = 0; i < runs; i++) {
            Matrix<fp32> A(rows1, cols1, ROW_WISE, CUDA);
            Matrix<fp32> B(rows2, cols2, ROW_WISE, CUDA);
            Matrix<fp32> G(rows1, cols2, ROW_WISE, CUDA);
    
            A.fill_random(i);
            B.fill_random(i + 1);
            mul_function(A, B, G);
            Matrix<fp32> C = mmul_cpu(A, B);
    
            auto diff = froebenius(C, G);
            if (diff > 1e-3) {
                auto diffs_ij = diffs(C, G);
                cout << "Differing indices: ";
                for (auto& p : diffs_ij) {
                    cout << "(" << p.first << ", " << p.second << ") ";
                }
                cout << endl;
            }
        }
    return 0;
}
