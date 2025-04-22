#include "include/Matrix.h"
#include <cassert>
#include <iostream>
#include <array>

using namespace Flux;

template<typename T, size_t Rows, size_t Cols>
void print_matrix(const Matrix<T, Rows, Cols>& m, const std::string& name) {
    std::cout << name << " =\n";
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            std::cout << m(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void test_matrix_constructors() {
    Matrix<double, 2, 2> m1;
    print_matrix(m1, "m1 (default)");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            assert(m1(i, j) == 0);

    Matrix<double, 2, 2> m2(3.14);
    print_matrix(m2, "m2 (filled with 3.14)");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            assert(m2(i, j) == 3.14);

    std::array<double, 4> arr = {1, 2, 3, 4};
    Matrix<double, 2, 2> m3(arr);
    print_matrix(m3, "m3 (from array {1,2,3,4})");
    assert(m3(0, 0) == 1 && m3(0, 1) == 2 && m3(1, 0) == 3 && m3(1, 1) == 4);
}

void test_matrix_ones_zeros() {
    auto ones = Matrix<double, 2, 2>::ones();
    print_matrix(ones, "ones");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            assert(ones(i, j) == 1);

    auto zeros = Matrix<double, 2, 2>::zeros();
    print_matrix(zeros, "zeros");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            assert(zeros(i, j) == 0);
}

void test_matrix_add_sub_scalar() {
    Matrix<double, 2, 2> m(2.0);
    print_matrix(m, "m (filled with 2.0)");
    auto m_add = m + 3.0;
    print_matrix(m_add, "m + 3.0");
    auto m_sub = m - 1.0;
    print_matrix(m_sub, "m - 1.0");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) {
            assert(m_add(i, j) == 5.0);
            assert(m_sub(i, j) == 1.0);
        }
}

void test_matrix_add_sub_matrix() {
    Matrix<double, 2, 2> a(2.0), b(3.0);
    print_matrix(a, "a (filled with 2.0)");
    print_matrix(b, "b (filled with 3.0)");
    auto c = a + b;
    print_matrix(c, "a + b");
    auto d = b - a;
    print_matrix(d, "b - a");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) {
            assert(c(i, j) == 5.0);
            assert(d(i, j) == 1.0);
        }
}

void test_matrix_mul_scalar() {
    Matrix<double, 2, 2> m(2.0);
    print_matrix(m, "m (filled with 2.0)");
    auto m_mul = m * 4.0;
    print_matrix(m_mul, "m * 4.0");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            assert(m_mul(i, j) == 8.0);
}

void test_matrix_transpose() {
    std::array<double, 4> arr = {1, 2, 3, 4};
    Matrix<double, 2, 2> m(arr);
    print_matrix(m, "m (original)");
    auto t = m.transpose();
    print_matrix(t, "transpose(m)");
    assert(t(0, 0) == 1 && t(0, 1) == 3 && t(1, 0) == 2 && t(1, 1) == 4);
}

void test_matrix_matmul() {
    std::array<double, 4> arrA = {1, 2, 3, 4};
    std::array<double, 4> arrB = {5, 6, 7, 8};
    Matrix<double, 2, 2> A(arrA), B(arrB);
    print_matrix(A, "A");
    print_matrix(B, "B");
    auto C = matmul(A, B);
    print_matrix(C, "A * B");
    assert(C(0, 0) == 19 && C(0, 1) == 22 && C(1, 0) == 43 && C(1, 1) == 50);
}

int main() {
    std::cout << "Testing Matrix constructors:\n";
    test_matrix_constructors();
    std::cout << "\nTesting Matrix ones/zeros:\n";
    test_matrix_ones_zeros();
    std::cout << "\nTesting Matrix add/sub with scalar:\n";
    test_matrix_add_sub_scalar();
    std::cout << "\nTesting Matrix add/sub with matrix:\n";
    test_matrix_add_sub_matrix();
    std::cout << "\nTesting Matrix multiply with scalar:\n";
    test_matrix_mul_scalar();
    std::cout << "\nTesting Matrix transpose:\n";
    test_matrix_transpose();
    std::cout << "\nTesting Matrix matmul:\n";
    test_matrix_matmul();
    std::cout << "\nAll Matrix tests passed!" << std::endl;
    return 0;
}