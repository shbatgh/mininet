#include "Value.h"
#include "../include/Matrix.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>

using namespace Flux;

template<typename T>
Value<T>::Value(std::shared_ptr<T> x) : data(std::move(x)), node(std::make_shared<GraphNode>()) {}

template<typename T>
Value<T>::Value(std::shared_ptr<T> x, std::string label) : data(std::move(x)), node(std::make_shared<GraphNode>()) {
    node->label = std::move(label);
}

template<typename T>
Value<T>::Value(std::shared_ptr<T> x, std::shared_ptr<GraphNode> node)
    : data(std::move(x)), node(std::move(node)) {}


template<typename T>
void Value<T>::backward() {
    std::vector<std::shared_ptr<GraphNode>> topo;
    std::unordered_set<GraphNode*> visited;
    std::function<void(std::shared_ptr<GraphNode>)> build_topo = [&](auto n) {
        if (visited.insert(n.get()).second) {
            for (const auto& wp : n->prev) {
                if (auto sp = wp.lock()) {
                    build_topo(sp);
                }
            }
            topo.push_back(n);
        }
    };
    build_topo(node);

    grad = T(1.0);
    std::reverse(topo.begin(), topo.end());
    for (auto node : topo) {
        if (!node->prev.empty()) {
            node->backward();
        }
    }
}

template<typename T>
ValuePtr<T> Value<T>::tanh() {
    T result = T();
    for (int i = 0; i < T::rows(); i++) {
        for (int j = 0; j < T::cols(); j++) {
            result(i, j) = std::tanh(this->data(i, j));
        }
    }
    auto out = std::make_shared<Value>(result, std::make_shared<GraphNode>("tanh", {this->shared_from_this(), nullptr}));

    out->node->backward = [out, self = this->shared_from_this(), result]() {
        self->grad += (1 - result * result) * out->grad;
    };

    return out;
}

template<typename T, size_t RowsA, size_t K, size_t ColsB>
ValuePtr<Matrix<T, RowsA, ColsB>> _matmul(const ValuePtr<Matrix<T, RowsA, K>>& lhs, const ValuePtr<Matrix<T, K, ColsB>>& rhs) {
    if (!lhs || !rhs) {
        std::cerr << "Error: matmul null ValuePtr." << std::endl;
        return nullptr;
    }

    auto data = std::make_shared<Matrix<T, RowsA, ColsB>>(Flux::matmul(*lhs->data, *rhs->data));
    auto out  = std::make_shared<Value<Matrix<T,RowsA,ColsB>>>(data);
    out->node->op = "matmul";
    out->node->prev = {lhs->node, rhs->node};

    // backward: dA = dC·Bᵀ, dB = Aᵀ·dC
    out->node->backward = [out, lhs, rhs]() {
        auto const& dC = out->grad;
        lhs->grad += Flux::matmul(dC, rhs->data->transpose());
        rhs->grad += Flux::matmul(lhs->data->transpose(), dC);
    };
    return out;
}


template<typename T>
ValuePtr<T> operator+(const ValuePtr<T>& lhs, const ValuePtr<T>& rhs) {
    if (!lhs || !rhs) {
        std::cerr << "Error: Adding null ValuePtr<T>." << std::endl;
        return nullptr;
    }
    auto result_data = std::make_shared<T>(*(lhs->data) + *(rhs->data));
    auto node = std::make_shared<GraphNode>("+", std::vector<std::weak_ptr<GraphNode>>{lhs->node, rhs->node});
    auto out = std::make_shared<Value<T>>(result_data, node);
    out->node->backward = [out, lhs, rhs]() {
        lhs->grad += out->grad;
        rhs->grad += out->grad;
    };

    return out;
}

template<typename T>
ValuePtr<T> operator-(const ValuePtr<T>& self) {
    if (!self) {
        std::cerr << "Error: Adding null ValuePtr<T>." << std::endl;
        return nullptr;
    }
    auto neg = std::make_shared<Value<T>>(std::make_shared<T>(-1.0));
    return self * neg;
}

template<typename T>
ValuePtr<T> operator-(const ValuePtr<T>& lhs, const ValuePtr<T>& rhs) {
    if (!lhs || !rhs) {
        std::cerr << "Error: Adding null ValuePtr<T>." << std::endl;
        return nullptr;
    }
    
    return lhs + (-rhs);
}

template<typename T>
ValuePtr<T> operator*(const ValuePtr<T>& lhs, const ValuePtr<T>& rhs) {
    if (!lhs || !rhs) {
        std::cerr << "Error: Multiplying null ValuePtr<T>." << std::endl;
        return nullptr;
    }
    auto result_data = std::make_shared<T>(*(lhs->data) * *(rhs->data));
    auto out = std::make_shared<Value<T>>(result_data);
    out->node->op = "*";
    out->node->prev = {lhs->node, rhs->node};
    out->node->backward = [out, lhs, rhs]() {
        lhs->grad += (*(rhs->data)) * out->grad;
        rhs->grad += (*(lhs->data)) * out->grad;
    };

    return out;
}

template<typename T>
ValuePtr<T> exp(const ValuePtr<T> &base) {
    auto out_data = std::make_shared<T>(base->data->exp());
    auto out = std::make_shared<Value<T>>(out_data);
    out->node->op = "exp";
    out->node->prev = {base->node};
    out->node->backward = [out, base]() {
        base->grad += *(out->data) * out->grad;
    };

    return out;
}

template<typename T>
ValuePtr<T> pow(const ValuePtr<T> &base, double exp) {
    auto out = std::make_shared<Value<T>>(std::make_shared<T>(base->data->pow(exp)));
    out->node->op = "pow";
    out->node->prev = {base->node};
    out->node->backward = [out, base, exp]() {
        base->grad += exp * base->data->pow(exp - 1) * out->grad;
    };

    return out;
}

template<typename T>
ValuePtr<T> operator/(const ValuePtr<T>& lhs, const ValuePtr<T>& rhs) {
    if (!lhs || !rhs) {
        std::cerr << "Error: Multiplying null ValuePtr<T>." << std::endl;
        return nullptr;
    }

    double neg = -1.0;
    return lhs * pow(rhs, neg);
}

int main() {

    auto x1 = std::make_shared<Value<Matrix<double,1,1>>>(std::make_shared<Matrix<double,1,1>>(2.0), "x1");
    auto x2 = std::make_shared<Value<Matrix<double,1,1>>>(std::make_shared<Matrix<double,1,1>>(0.0), "x2");
    auto w1 = std::make_shared<Value<Matrix<double,1,1>>>(std::make_shared<Matrix<double,1,1>>(-3.0), "w1");
    auto w2 = std::make_shared<Value<Matrix<double,1,1>>>(std::make_shared<Matrix<double,1,1>>(1.0), "w2");
    auto b = std::make_shared<Value<Matrix<double,1,1>>>(std::make_shared<Matrix<double,1,1>>(6.8813735870195432), "b");
    auto x1w1 = x1 * w1; x1w1->node->label = "x1*w1";
    auto x2w2 = x2 * w2; x2w2->node->label = "x2*w2";
    auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2->node->label = "x1*w1 + x2*w2";
    auto n = x1w1x2w2 + b; n->node->label = "n";
    auto e = exp(2*n); e->node->label = "e";
    auto o = (e - 1) / (e + 1); o->node->label = "o";

    o->backward();

    std::cout << "x1: " << x1 << std::endl;
    std::cout << "w1: " << w1 << std::endl;
    std::cout << "x2: " << x2 << std::endl;
    std::cout << "w2: " << w2 << std::endl;

    std::cout << "x1*w1: " << x1w1 << std::endl;
    std::cout << "x2*w2: " << x2w2 << std::endl;

    std::cout << "x1*w2 + x2*w2: " << x1w1x2w2 << std::endl;
    std::cout << "b: " << b << std::endl;

    std::cout << "n: " << n << std::endl;

    std::cout << "o: " << o << std::endl;

    return 0;
}