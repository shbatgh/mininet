#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <array>

namespace Flux {

template<typename T, size_t Rows, size_t Cols>
class Matrix {
protected:
    std::unique_ptr<T[]> m_data;

public:
    ~Matrix() = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    /*Matrix(const Matrix& o)
      : m_data(std::make_unique<T[]>(Rows*Cols))
    {
      std::copy(o.m_data.get(),
                o.m_data.get() + Rows*Cols,
                m_data.get());
    }

    Matrix& operator=(const Matrix& o)
    {
      if (this != &o) {
        std::copy(o.m_data.get(),
                  o.m_data.get() + Rows*Cols,
                  m_data.get());
      }
      return *this;
    }*/

    Matrix() : m_data(std::make_unique<T[]>(Rows * Cols)) {}
    Matrix(const std::array<T, Rows * Cols>& data) : m_data(std::make_unique<T[]>(Rows * Cols)) {
        for (int i = 0; i < Rows * Cols; i++) {
            m_data[i] = data[i];
        }
    }
    template<typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    Matrix(Scalar s) : m_data(std::make_unique<T[]>(Rows * Cols)) {
        for (int i = 0; i < Rows * Cols; i++) {
            m_data[i] = s;
        }
    }

    static constexpr size_t rows() noexcept { return Rows; }
    static constexpr size_t cols() noexcept { return Cols; }
    static constexpr size_t size() noexcept { return Rows * Cols; }
    
    T* data() noexcept { return m_data.data(); }
    const T* data() const noexcept { return m_data.data(); }


    static constexpr Matrix ones() {
        return Matrix<T, Rows, Cols>(1);
    }

    static constexpr Matrix zeros() {
        return Matrix<T, Rows, Cols>(0);
    }

    static Matrix rand(double lbound = -1.0, double rbound = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(lbound, rbound);

        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = static_cast<T>(dis(gen));
            }
        }
        return result;
    }
    

    constexpr T& operator()(size_t i, size_t j) { return m_data[i*Cols + j]; }
    constexpr const T& operator()(size_t i, size_t j) const { return m_data[i*Cols + j]; }

    template<typename U>
    Matrix<T, Rows, Cols> operator+(const Matrix<U, Rows, Cols>& rhs) const {
        //if (Matrix<T, Rows, Cols>::size() != Matrix<U, Rows, Cols>::size()) throw std::runtime_error("Vector size mismatch");
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows * Cols; i++) {
            result.m_data[i] = m_data[i] + rhs.m_data[i];
        }

        return result;
    }

    template<typename U>
    Matrix<T, Rows, Cols>& operator+=(const Matrix<U, Rows, Cols>& rhs) {
        *this = *this + rhs;
        return *this;
    }

    template<typename Num, typename = std::enable_if_t<std::is_arithmetic_v<Num>>>
    Matrix operator+(const Num& scalar) const {
        Matrix<Num, Rows, Cols> result;
        for (int i = 0; i < Rows * Cols; i++) {
            result.m_data[i] = m_data[i] + scalar;
        }

        return result;
    }

    template<typename U>
    Matrix<T, Rows, Cols> operator-(const Matrix<U, Rows, Cols>& rhs) const {
        //if (Matrix<T, Rows, Cols>::size() != Matrix<U, Rows, Cols>::size()) throw std::runtime_error("Vector size mismatch");
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows * Cols; i++) {
            result.m_data[i] = m_data[i] - rhs.m_data[i];
        }

        return result;
    }

    template<typename Num, typename = std::enable_if_t<std::is_arithmetic_v<Num>>>
    Matrix<T, Rows, Cols> operator-(const Num& scalar) const {
        Matrix<Num, Rows, Cols> result;
        for (int i = 0; i < Rows * Cols; i++) {
            result.m_data[i] = m_data[i] - scalar;
        }

        return result;
    }

    template<typename U>
    Matrix<T, Rows, Cols> operator*(const Matrix<U, Rows, Cols>& rhs) const {
        Matrix<T, Rows, Cols> result;
        for (int i = 0; i < Rows * Cols; i++) {
            result.m_data[i] = m_data[i] * rhs.m_data[i];
        }

        return result;
    }

    Matrix<T, Rows, Cols> pow(double p) const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; i++) {
            for (size_t j = 0; j < Cols; j++) {
                result(i, j) = std::pow(this->operator()(i, j), p);
            }
        }

        return result;
    }

    Matrix<T, Rows, Cols> exp() const {
        Matrix<T, Rows, Cols> result;
        for (size_t i = 0; i < Rows; i++) {
            for (size_t j = 0; j < Cols; j++) {
                result(i, j) = std::exp(this->operator()(i, j));
            }
        }

        return result;
    }

    Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> result;
        for (int r = 0; r < Rows; r++) {
            for (int c = 0; c < Cols; c++) {
                result(c, r) = (*this)(r, c);
            }
        }

        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T, Rows, Cols>& m) {
        for (size_t r = 0; r < Rows; r++) {
            for (size_t c = 0; c < Cols; c++) {
                os << m(r, c) << " ";
            }
            if (r < Rows - 1) {
                os << std::endl;
            }
        } 

        return os; 
    }

    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Matrix<T, Rows, Cols>>& m) {
        return os << *m;
    }
};

template<typename T, size_t Rows, size_t Cols, typename Num, typename = std::enable_if_t<std::is_arithmetic_v<Num>>>
Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, Cols>& lhs, const Num& scalar) {
    Matrix<Num, Rows, Cols> result;
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            result(i, j) = lhs(i, j) * scalar;
        }
    }

    return result;
}

template<typename T, size_t Rows, size_t Cols, typename Num, typename = std::enable_if_t<std::is_arithmetic_v<Num>>>
Matrix<T, Rows, Cols> operator*(const Num& scalar, const Matrix<T, Rows, Cols>& rhs) {
    Matrix<Num, Rows, Cols> result;
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            result(i, j) = rhs(i, j) * scalar;
        }
    }

    return result;
}

template<typename T, size_t RowsA, size_t K, size_t ColsB>
Matrix<T, RowsA, ColsB> matmul(const Matrix<T, RowsA, K>& A, const Matrix<T, K, ColsB>& B) {
    Matrix<T, RowsA, ColsB> result;
    for (size_t row = 0; row < RowsA; row++) {
        for (size_t col = 0; col < ColsB; col++) {
            T sum = T();
            for (size_t k = 0; k < K; k++) {
                sum += A(row, k) * B(k, col);
            }
            result(row, col) = sum;
        }
    }

    return result;
}

}

#endif