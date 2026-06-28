#pragma once

#include "spareribs/core/concepts.hpp"
#include "spareribs/internal/vec2.hpp"

#include <cassert>
#include <concepts>
#include <type_traits>

namespace spareribs {

namespace model {

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

} // namespace model

namespace integrator {

template <typename TModel>
class RungeKutta4 {
    TModel model_;

  public:
    RungeKutta4(TModel model) : model_{model} {}

    template <concepts::gpu_float F>
        requires std::is_invocable_r_v<Vec2<F>, TModel, Vec2<F>>
    __host__ __device__ Vec2<F> operator()(Vec2<F> y, F h) const {
        const Vec2<F> k1 = model_(y), k2 = model_(y + k1 * h / F{2}),
                      k3 = model_(y + k2 * h / F{2}), k4 = model_(y + k3 * h);
        return y + (k1 + F{2} * k2 + F{2} * k3 + k4) * h / F{6};
    }
};

} // namespace integrator

} // namespace spareribs
