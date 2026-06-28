#pragma once

#include "spareribs/core/simulation_pack.hpp"
#include "spareribs/internal/div_ceil.hpp"
#include "spareribs/internal/vec2.hpp"

#include <cassert>
#include <concepts>
#include <cuda.h>
#include <type_traits>

namespace spareribs::evolution_map {

namespace detail::independent {

template <typename TIntegrator, std::floating_point F>
    requires std::is_trivially_copyable_v<TIntegrator> and
             std::is_invocable_r_v<Vec2<F>, TIntegrator, Vec2<F>, F>
__device__ void kernel_impl(TIntegrator integrator, F* first_out, F* second_out, F const* first_in,
                            F const* second_in, F step, std::size_t len) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);
    const unsigned int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        Vec2 const out = integrator(Vec2{first_in[g_idx], second_in[g_idx]}, step);
        first_out[g_idx] = out.first;
        second_out[g_idx] = out.second;
    }
}

template <typename TIntegrator, std::floating_point F>
    requires std::is_trivially_copyable_v<TIntegrator> and
             std::is_invocable_r_v<Vec2<F>, TIntegrator, Vec2<F>, F>
__global__ void kernel_in_place(TIntegrator integrator, F* __restrict__ first,
                                F* __restrict__ second, F step, std::size_t len) {
    kernel_impl(integrator, first, second, first, second, step, len);
}

template <typename TIntegrator, std::floating_point F>
    requires std::is_trivially_copyable_v<TIntegrator> and
             std::is_invocable_r_v<Vec2<F>, TIntegrator, Vec2<F>, F>
__global__ void kernel_out_of_place(TIntegrator integrator, F* __restrict__ first_out,
                                    F* __restrict__ second_out, F const* __restrict__ first_in,
                                    F const* __restrict__ second_in, F step, std::size_t len) {
    kernel_impl(integrator, first_out, second_out, first_in, second_in, step, len);
}

} // namespace detail::independent

template <typename TIntegrator>
    requires std::is_trivially_copyable_v<TIntegrator>
class Independent {
    TIntegrator integrator_;
    unsigned int block_threads_;

  public:
    Independent(TIntegrator integrator, unsigned int block_threads)
        : integrator_{integrator}, block_threads_{block_threads} {}

    template <std::floating_point F>
        requires std::is_invocable_r_v<Vec2<F>, TIntegrator, Vec2<F>, F>
    void evolve_in_place(SimulationPack<F>& pack, F step) {
        unsigned int const grid_blocks = div_ceil(pack.len(), block_threads_);

        detail::independent::kernel_in_place<<<grid_blocks, block_threads_>>>(
            integrator_, pack.u(), pack.v(), step, pack.len());
    }

    template <std::floating_point F>
        requires std::is_invocable_r_v<Vec2<F>, TIntegrator, Vec2<F>, F>
    void evolve_out_of_place(SimulationPack<F>& out_pack, SimulationPack<F> const& in_pack,
                             F step) {
        assert(out_pack.len() == in_pack.len());
        unsigned int const grid_blocks = div_ceil(in_pack.len(), block_threads_);

        detail::independent::kernel_out_of_place<<<grid_blocks, block_threads_>>>(
            integrator_, out_pack.u(), out_pack.v(), in_pack.u(), in_pack.v(), step, in_pack.len());
    }
};

} // namespace spareribs::evolution_map
