
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <map>
#include <functional>

#include "matrix.cuh"
#include "kernels.cuh"
#include "test_class_matrix.cu"

using std::cout;
using std::endl;
using std::cin;
using std::vector;
using std::string;
using std::map;
using std::function;

#ifndef RUNS_NUM
#define RUNS_NUM 4
#endif

enum class FillType {
    RANDOM,
    ONES,
    ZEROS
};

typedef function<void(Matrix<fp32>&, Matrix<fp32>&, Matrix<fp32>&)> KernelFunc;

template <typename T>
T calculate_max_diff(Matrix<T>& A, Matrix<T>& B) {
    A.cpu();
    B.cpu();
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    T max_diff = 0.0;
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < A.cols(); j++) {
            T diff = std::abs(A.get(i, j) - B.get(i, j));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

template <typename T>
Matrix<T> mmul_cpu(Matrix<T>& A, Matrix<T>& B) {
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

template <typename T>
void print_heatmap(Matrix<T>& GPU_C, Matrix<T>& CPU_C, T precision) {
    size_t rows = GPU_C.rows();
    size_t cols = GPU_C.cols();
    size_t grid_r = std::min(rows, (size_t)32);
    size_t grid_c = std::min(cols, (size_t)32);
    size_t step_r = rows / grid_r;
    size_t step_c = cols / grid_c;

    cout << "\nError Heatmap (" << grid_r << "x" << grid_c << " sampling):" << endl;
    for (size_t i = 0; i < grid_r; i++) {
        for (size_t j = 0; j < grid_c; j++) {
            bool has_error = false;
            for (size_t bi = i * step_r; bi < (i + 1) * step_r; bi++) {
                for (size_t bj = j * step_c; bj < (j + 1) * step_c; bj++) {
                    if (std::abs(GPU_C.get(bi, bj) - CPU_C.get(bi, bj)) > precision) {
                        has_error = true; break;
                    }
                }
                if (has_error) break;
            }
            cout << (has_error ? "X" : ".");
        }
        cout << endl;
    }
}

void verify_result(Matrix<fp32>& GPU_C, Matrix<fp32>& CPU_C, fp32 precision = 1e-4) {
    fp32 max_diff = calculate_max_diff(GPU_C, CPU_C);
    if (max_diff > precision) {
        cout << "[FAILED] Max difference: " << std::scientific << max_diff << endl;
        print_heatmap(GPU_C, CPU_C, precision);
    } else {
        cout << "[PASSED] Max difference: " << std::scientific << max_diff << endl;
    }
}

template <typename T>
void fill_matrix(Matrix<T>& M, FillType type, unsigned long long seed) {
    if (type == FillType::RANDOM) {
        M.fill_random(seed);
    } else {
        M.cpu();
        M.to_layout(ROW_WISE);
        for (size_t i = 0; i < M.rows(); i++)
            for (size_t j = 0; j < M.cols(); j++)
                M.set(i, j, (type == FillType::ONES ? 1.0f : 0.0f));
        M.cuda();
    }
}

void run_test(KernelFunc kernel, size_t N, size_t K, size_t M, FillType fill, int runs = RUNS_NUM) {
    for (int i = 0; i < runs; i++) {
        Matrix<fp32> A(N, K, ROW_WISE, CUDA);
        Matrix<fp32> B(K, M, ROW_WISE, CUDA);
        Matrix<fp32> G(N, M, ROW_WISE, CUDA);
        fill_matrix(A, fill, (unsigned long long)i);
        fill_matrix(B, fill, (unsigned long long)i + 1337);

        kernel(A, B, G);

        A.cpu(); B.cpu();
        Matrix<fp32> C = mmul_cpu(A, B);
        verify_result(G, C);
    }
}

void iterative_stress_test(KernelFunc kernel) {
    for (size_t size = 16; size <= 1024; size *= 2) {
        cout << "\n--- Size: " << size << "x" << size << " ---" << endl;
        run_test(kernel, size, size, size, FillType::RANDOM, 1);
    }
}

void menu() {
    map<string, KernelFunc> kernel_registry = {
        {"Simple", simple_launcher},
        {"Blocked", blocked_launcher}
    };

    while (true) {
        cout << "\n=== CuTesseract Test CLI ===" << endl;
        cout << "1. Run Class Matrix Tests" << endl;
        cout << "2. Standard Kernel Run (512x512)" << endl;
        cout << "3. Iterative Stress Test (16->1024)" << endl;
        cout << "4. Exit" << endl;
        cout << "Choice: ";

        int choice;
        if (!(cin >> choice)) break;

        if (choice == 1) {
            test_layout();
            test_layout_switch();
        } else if (choice == 2 || choice == 3) {
            cout << "\nSelect Kernel:" << endl;
            int idx = 1;
            vector<string> names;
            for (auto const& [name, func] : kernel_registry) {
                cout << idx++ << ". " << name << endl;
                names.push_back(name);
            }
            int k_choice; cin >> k_choice;
            if (k_choice < 1 || k_choice > names.size()) continue;
            KernelFunc kernel = kernel_registry[names[k_choice - 1]];

            cout << "Select Fill:\n1. Random\n2. Ones\n3. Zeros\nChoice: ";
            int f_choice; cin >> f_choice;
            FillType fill = (f_choice == 2 ? FillType::ONES : (f_choice == 3 ? FillType::ZEROS : FillType::RANDOM));

            if (choice == 2) run_test(kernel, 512, 512, 512, fill);
            else iterative_stress_test(kernel);
        } else if (choice == 4) break;
    }
}

int main() {
    menu();
    return 0;
}
