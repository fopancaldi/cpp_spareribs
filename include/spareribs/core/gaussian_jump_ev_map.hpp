#pragma once

#include "spareribs/core/concepts.hpp"
#include "spareribs/core/simulation_pack.hpp"
#include "spareribs/internal/curand_generate.hpp"
#include "spareribs/internal/div_ceil.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cuda.h>
#include <curand.h>

namespace spareribs::evolution_map {

namespace detail::gaussian_jump {

template <concepts::gpu_float F>
__device__ void kernel_impl(F* arr_out, F const* arr_in, F const* __restrict__ random_vals,
                            std::size_t len) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    assert(blockIdx.y == 0u and blockIdx.z == 0u);

    unsigned int const g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < len) {
        arr_out[g_idx] = arr_in[g_idx] + random_vals[g_idx];
    }
}

template <concepts::gpu_float F>
__global__ void kernel_in_place(F* __restrict__ arr, F const* __restrict__ random_vals,
                                std::size_t len) {
    kernel_impl(arr, arr, random_vals, len);
}

template <concepts::gpu_float F>
__global__ void kernel_out_of_place(F* __restrict__ arr_out, F* __restrict__ arr_in,
                                    F const* __restrict__ random_vals, std::size_t len) {
    kernel_impl(arr_out, arr_in, random_vals, len);
}

} // namespace detail::gaussian_jump

template <concepts::gpu_float F>
class GaussianJump {
    F temperature_;
    F* random_vals_;
    std::size_t len_;
    curandGenerator_t generator_;
    unsigned int block_threads_;

  public:
    GaussianJump(F temperature, std::size_t len, std::size_t seed, unsigned int block_threads)
        : temperature_{temperature}, random_vals_{nullptr}, len_{len}, generator_{nullptr},
          block_threads_{block_threads} {
        cudaError_t e = cudaMalloc(&random_vals_, len_ * sizeof(F));
        assert(e == cudaSuccess);
        curandStatus_t s = curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
        assert(s == CURAND_STATUS_SUCCESS);
        s = curandSetPseudoRandomGeneratorSeed(generator_, seed);
        assert(s == CURAND_STATUS_SUCCESS);
        assert(len_ % 2 == 0);
    }
    GaussianJump(GaussianJump const& other)
        : GaussianJump(other.temperature_, other.len_, 0, other.block_threads_) {
        cudaMemcpy(random_vals_, other.random_vals_, len_ * sizeof(F), cudaMemcpyHostToDevice);
    }
    GaussianJump(GaussianJump&& other) = delete;
    ~GaussianJump() {
        cudaError_t e = cudaFree(random_vals_);
        assert(e == cudaSuccess);
        curandStatus_t s = curandDestroyGenerator(generator_);
        assert(s == CURAND_STATUS_SUCCESS);
    }

    void evolve_in_place(SimulationPack<F>& pack, F step) {
        assert(pack.len() == len_);
        curandStatus_t const s = generate_normal(generator_, random_vals_, len_, F{0},
                                                 std::sqrt(F{2} * temperature_ * step));
        assert(s == CURAND_STATUS_SUCCESS);
        unsigned int const grid_blocks = div_ceil(pack.len(), block_threads_);

        detail::gaussian_jump::kernel_in_place<<<grid_blocks, block_threads_>>>(
            pack.u(), random_vals_, pack.len());
    }
    void evolve_out_of_place(SimulationPack<F>& pack_out, SimulationPack<F>& pack_in, F step) {
        assert(pack_in.len() == len_);
        assert(pack_out.len() == len_);
        curandStatus_t const s = generate_normal(generator_, random_vals_, len_, F{0},
                                                 std::sqrt(F{2} * temperature_ * step));
        assert(s == CURAND_STATUS_SUCCESS);
        unsigned int const grid_blocks = div_ceil(pack_in.len(), block_threads_);

        detail::gaussian_jump::kernel_out_of_place<<<grid_blocks, block_threads_>>>(
            pack_out.u(), pack_in.u(), random_vals_, pack_in.len());
    }
};

} // namespace spareribs::evolution_map
