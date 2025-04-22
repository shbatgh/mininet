#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <memory>
#include <functional>
#include <type_traits>
#include <string>
#include <vector>
#include <cmath>
#include <initializer_list>
#include "../include/Matrix.h"

using Flux::Matrix;

struct GraphNode {
    std::vector<std::weak_ptr<GraphNode>> prev;
    std::function<void()> backward;
    std::string op;
    std::string label;

    GraphNode(std::string op, std::initializer_list<std::weak_ptr<GraphNode>> prev = {}, std::function<void()> backward = [](){}, std::string label = "") 
        : op(op), prev(prev), backward(backward), label(label) {}
    GraphNode() 
        : op(""), prev(), backward([](){}), label("") {}
    GraphNode(const std::string &op, const std::vector<std::weak_ptr<GraphNode>> &prev)
    : op(op), prev(prev), backward([](){}), label("") {}
};

template<typename T>
class Value;

// Shared‐pointer alias
template<typename T>
using ValuePtr = std::shared_ptr<Value<T>>;

template<typename T>
class Value : public std::enable_shared_from_this<Value<T>> {
public:
    std::shared_ptr<T> data;
    mutable T grad = T();
    std::shared_ptr<GraphNode> node;

    /// Construct a leaf node (no parents)
    explicit Value(std::shared_ptr<T> x);
    Value(std::shared_ptr<T> x, std::string label);
    Value(std::shared_ptr<T> x, std::shared_ptr<GraphNode> node);

    void backward();
    
    ValuePtr<T> tanh();
};

// Binary operators
template<typename T, size_t RowsA, size_t K, size_t ColsB>
ValuePtr<T> matmul(const ValuePtr<Matrix<T, RowsA, K>>& lhs, const ValuePtr<Matrix<T, K, ColsB>>& rhs);

template<typename T>
ValuePtr<T> operator+(const ValuePtr<T>& a, const ValuePtr<T>& b);

template<typename T>
ValuePtr<T> operator-(const ValuePtr<T>& a, const ValuePtr<T>& b);

template<typename T>
ValuePtr<T> operator*(const ValuePtr<T>& a, const ValuePtr<T>& b);

template<typename T>
ValuePtr<T> operator/(const ValuePtr<T>& a, const ValuePtr<T>& b);

// Unary negation
template<typename T>
ValuePtr<T> operator-(const ValuePtr<T>& a);

// Exponential and power
template<typename T>
ValuePtr<T> exp(const ValuePtr<T>& a);

template<typename T>
ValuePtr<T> pow(const ValuePtr<T>& a, double exponent);

template<typename Scalar, size_t R, size_t K, size_t C>
ValuePtr<Flux::Matrix<Scalar,R,C>> matmul(const ValuePtr<Flux::Matrix<Scalar,R,K>>& lhs, const ValuePtr<Flux::Matrix<Scalar,K,C>>& rhs);

// Scalar‐ValuePtr overloads
template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator+(const ValuePtr<T>& v, Scalar x) {
    return v + std::make_shared<Value<T>>(std::make_shared<T>(x));
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator+(Scalar x, const ValuePtr<T>& v) {
    return std::make_shared<Value<T>>(std::make_shared<T>(x)) + v;
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator-(const ValuePtr<T>& v, Scalar x) {
    return v - std::make_shared<Value<T>>(std::make_shared<T>(x));
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator-(Scalar x, const ValuePtr<T>& v) {
    return std::make_shared<Value<T>>(std::make_shared<T>(x)) - v;
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator*(const ValuePtr<T>& v, Scalar x) {
    return v * std::make_shared<Value<T>>(std::make_shared<T>(x));
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator*(Scalar x, const ValuePtr<T>& v) {
    return std::make_shared<Value<T>>(std::make_shared<T>(x)) * v;
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator/(const ValuePtr<T>& v, Scalar x) {
    return v / std::make_shared<Value<T>>(std::make_shared<T>(x));
}

template<typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
ValuePtr<T> operator/(Scalar x, const ValuePtr<T>& v) {
    return std::make_shared<Value<T>>(std::make_shared<T>(x)) / v;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
    os << "Value(data=" << v.data << ", grad=" << v.grad;
    if (v.node && !v.node->op.empty()) os << ", op=" << v.node->op;
    os << ")";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const ValuePtr<T>& v) {
    return os << *v;
}

#endif