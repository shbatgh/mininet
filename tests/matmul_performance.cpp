#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>                 
#include "../include/Matrix.h"

using namespace Flux;

int main() {
    constexpr size_t N = 4000;

    // 1) Correctness check: I * B == B
    auto I = std::make_unique<Matrix<double, N, N>>(0.0);
    for (size_t i = 0; i < N; ++i) {
        (*I)(i, i) = 1.0;
    }
    auto B = std::make_unique<Matrix<double, N, N>>(Matrix<double, N, N>::rand());
    auto C = matmul(*I, *B);

    double max_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            max_diff = std::max(max_diff, std::abs(C(i, j) - (*B)(i, j)));
        }
    }
    std::cout << "Max abs diff (I*B vs B): " << max_diff << "\n";

    // 2) Performance timing
    auto A = std::make_unique<Matrix<double, N, N>>(Matrix<double, N, N>::rand());
    auto D = std::make_unique<Matrix<double, N, N>>(Matrix<double, N, N>::rand());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto E  = matmul(*A, *D);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Time for " << N << "x" << N << " multiply: "
              << dt.count() << " seconds\n";

    return (max_diff < 1e-9) ? 0 : 1;
}