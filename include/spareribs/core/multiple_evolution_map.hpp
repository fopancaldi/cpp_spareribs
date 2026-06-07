#pragma once

#include "spareribs/core/util.hpp"
#include "spareribs/core/vec2.hpp"

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cuda.h>
#include <curand.h>
#include <type_traits>

namespace spareribs {

namespace detail::independent_ev_map {

template <typename TSingleEvMap, std::floating_point F>
    requires std::is_trivially_copyable_v<TSingleEvMap> and
             std::is_invocable_r_v<Vec2<F>, TSingleEvMap, Vec2<F>, F>
static __global__ void kernel_in_place(TSingleEvMap single_em, F* first, F* second, F step,
                                       std::size_t len) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    assert(blockIdx.y == 0u and blockIdx.z == 0u);
    const unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        const Vec2<F> out = single_em(Vec2<F>{first[g_idx], second[g_idx]}, step);
        first[g_idx] = out.first;
        second[g_idx] = out.second;
    }
}

template <typename TSingleEvMap, std::floating_point F>
    requires std::is_trivially_copyable_v<TSingleEvMap> and
             std::is_invocable_r_v<Vec2<F>, TSingleEvMap, Vec2<F>, F>
static __global__ void kernel_out_of_place(TSingleEvMap single_em, F const* __restrict__ first_in,
                                           F const* __restrict__ second_in,
                                           F* __restrict__ first_out, F* __restrict__ second_out,
                                           F step, std::size_t len) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    assert(blockIdx.y == 0u and blockIdx.z == 0u);
    const unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        const Vec2<F> out = single_em(Vec2<F>{first_in[g_idx], second_in[g_idx]}, step);
        first_out[g_idx] = out.first;
        second_out[g_idx] = out.second;
    }
}

} // namespace detail::independent_ev_map

template <typename TSingleEvMap, std::floating_point F>
    requires std::is_trivially_copyable_v<TSingleEvMap> and
             std::is_invocable_r_v<Vec2<F>, TSingleEvMap, Vec2<F>, F>
class IndependentEvMap {
    TSingleEvMap single_ev_map_;
    unsigned int block_threads_;

  public:
    IndependentEvMap(TSingleEvMap single_ev_map, unsigned int block_threads, F)
        : single_ev_map_{single_ev_map}, block_threads_{block_threads} {}
    void evolve_in_place(F* first, F* second, F step, std::size_t len) {
        const unsigned int grid_blocks = div_ceil(len, block_threads_);
        detail::independent_ev_map::kernel_in_place<<<grid_blocks, block_threads_>>>(
            single_ev_map_, first, second, step, len);
    }
    void evolve_out_of_place(F const* __restrict__ first_in, F const* __restrict__ second_in,
                             F* __restrict__ first_out, F* __restrict__ second_out, F step,
                             std::size_t len) {
        const unsigned int grid_blocks = div_ceil(len, block_threads_);
        detail::independent_ev_map::kernel_out_of_place<<<grid_blocks, block_threads_>>>(
            single_ev_map_, first_in, second_in, first_out, second_out, step, len);
    }
    TSingleEvMap& single_ev_map() { return single_ev_map_; }
};

template <typename TEvMap1, typename TEvMap2, std::floating_point F>
    requires requires(TEvMap1 em1, TEvMap2 em2, F f, F* p, F const* cp, std::size_t s) {
        { em1.evolve_in_place(p, p, f, s) } -> std::same_as<void>;
        { em1.evolve_out_of_place(p, p, cp, cp, f, s) } -> std::same_as<void>;
        { em2.evolve_in_place(p, p, f, s) } -> std::same_as<void>;
        { em2.evolve_out_of_place(p, p, cp, cp, f, s) } -> std::same_as<void>;
    }
class RibsCompositionEvMap {
    TEvMap1 ev_map_1_;
    TEvMap2 ev_map_2_;

  public:
    RibsCompositionEvMap(TEvMap1&& ev_map_1, TEvMap2&& ev_map_2, F)
        : ev_map_1_{ev_map_1_}, ev_map_2_{ev_map_2_} {}
    void evolve_in_place(F* first, F* second, F step, std::size_t len) {
        ev_map_1_.evolve_in_place(first, second, step / F{2}, len);
        ev_map_2_.evolve_in_place(first, second, step, len);
        ev_map_1_.evolve_in_place(first, second, step / F{2}, len);
    }
    void evolve_out_of_place(F const* __restrict__ first_in, F const* __restrict__ second_in,
                             F* __restrict__ first_out, F* __restrict__ second_out, F step,
                             std::size_t len) {
        ev_map_1_.evolve_out_of_place(first_in, second_in, first_out, second_out, step / F{2}, len);
        ev_map_2_.evolve_in_place(first_out, second_out, step, len);
        ev_map_1_.evolve_in_place(first_out, second_out, step / F{2}, len);
    }
};

namespace detail::gaussian_jump_ev_map {

template <std::floating_point F>
static __global__ void kernel_in_place(F const* __restrict__ random_vals, F* arr, std::size_t len) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    assert(blockIdx.y == 0u and blockIdx.z == 0u);
    const unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        arr[g_idx] += random_vals[g_idx];
    }
}

template <std::floating_point F>
static __global__ void kernel_out_of_place(F const* __restrict__ random_vals,
                                           F const* __restrict__ arr_in, F* __restrict__ arr_out,
                                           std::size_t len) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    assert(blockIdx.y == 0u and blockIdx.z == 0u);
    const unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        arr_out[g_idx] = arr_in[g_idx] + random_vals[g_idx];
    }
}

} // namespace detail::gaussian_jump_ev_map

template <std::floating_point F>
class GaussianJumpEvMap {
    F temperature_;
    F* random_vals_;
    std::size_t len_;
    curandGenerator_t generator_;
    unsigned int block_threads_;

  public:
    GaussianJumpEvMap(F temperature, std::size_t len, std::size_t seed, unsigned int block_threads)
        : temperature_{temperature}, random_vals_{nullptr}, len_{len}, generator_{},
          block_threads_{block_threads} {
        cudaError_t e = cudaMalloc(&random_vals_, len_ * sizeof(F));
        assert(e == cudaSuccess);
        curandStatus_t s = curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
        assert(s == CURAND_STATUS_SUCCESS);
        s = curandSetPseudoRandomGeneratorSeed(generator_, seed);
        assert(s == CURAND_STATUS_SUCCESS);
        assert(len_ % 2 == 0);
    }
    void evolve_in_place(F* first, F*, F, std::size_t len) {
        assert(len == len_);
        // TODO: Extend to F == double (requires using curandGenerateNormalDouble!)
        static_assert(std::is_same_v<F, float>);
        curandStatus_t const s = curandGenerateNormal(generator_, random_vals_, len_, F{0},
                                                      std::sqrt(F{2} * temperature_));
        assert(s == CURAND_STATUS_SUCCESS);
        const unsigned int grid_blocks = div_ceil(len, block_threads_);
        detail::gaussian_jump_ev_map::kernel_in_place<<<grid_blocks, block_threads_>>>(random_vals_,
                                                                                       first, len_);
    }
    void evolve_out_of_place(F const* __restrict__ first_in, F const*, F* __restrict__ first_out,
                             F*, F, std::size_t len) {
        assert(len == len_);
        curandStatus_t const s = curandGenerateNormal(generator_, random_vals_, len_, F{0},
                                                      std::sqrt(F{2} * temperature_));
        assert(s == CURAND_STATUS_SUCCESS);
        const unsigned int grid_blocks = div_ceil(len, block_threads_);
        detail::gaussian_jump_ev_map::kernel_out_of_place<<<grid_blocks, block_threads_>>>(
            random_vals_, first_in, first_out, len_);
    }
    ~GaussianJumpEvMap() {
        cudaFree(random_vals_);
        curandDestroyGenerator(generator_);
    }
};

} // namespace spareribs
