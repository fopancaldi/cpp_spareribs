#pragma once

#include "spareribs/core/vec2.hpp"

#include <cassert>
#include <concepts>
#include <type_traits>

namespace spareribs {

template <std::floating_point F>
struct FhnNeuron {
    F epsilon;
    F a;
    __host__ __device__ Vec2<F> operator()(Vec2<F> vec) const {
        const auto [u, v] = vec;
        return Vec2{u - u * u * u / F{3} - v, epsilon * (u + a)};
    }
};

template <std::floating_point F>
struct HarmonicOscillator {
    F m;
    F omega;
    __host__ __device__ Vec2<F> operator()(Vec2<F> vec) const {
        const auto [q, p] = vec;
        return Vec2{p / m, -m * omega * omega * q};
    }
};

template <typename TFunction, std::floating_point F>
    requires std::is_invocable_r_v<Vec2<F>, TFunction, Vec2<F>>
class Rk4EvMap {
    TFunction function_;

  public:
    Rk4EvMap(TFunction function, F) : function_{function} {}
    __host__ __device__ Vec2<F> operator()(Vec2<F> y, F h) const {
        const Vec2<F> k1 = function_(y), k2 = function_(y + k1 * h / F{2}),
                      k3 = function_(y + k2 * h / F{2}), k4 = function_(y + k3 * h);
        return y + (k1 + F{2} * k2 + F{2} * k3 + k4) * h / F{6};
    }
    TFunction& function() { return function_; }
};

} // namespace spareribs
