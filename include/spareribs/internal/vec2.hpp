#pragma once

#include <cassert>
#include <concepts>

namespace spareribs {

template <std::floating_point F>
struct Vec2 {
    F first;
    F second;
};

template <std::floating_point F>
__host__ __device__ Vec2<F> operator+(Vec2<F> lhs, Vec2<F> rhs) {
    return {lhs.first + rhs.first, lhs.second + rhs.second};
}

template <std::floating_point F>
__host__ __device__ Vec2<F> operator*(F f, Vec2<F> v) {
    return {f * v.first, f * v.second};
}

template <std::floating_point F>
__host__ __device__ Vec2<F> operator*(Vec2<F> v, F f) {
    return f * v;
}

template <std::floating_point F>
__host__ __device__ Vec2<F> operator-(Vec2<F> lhs, Vec2<F> rhs) {
    return lhs + F{-1} * rhs;
}

template <std::floating_point F>
__host__ __device__ Vec2<F> operator/(Vec2<F> lhs, F rhs) {
    return {lhs.first / rhs, lhs.second / rhs};
}

} // namespace spareribs
