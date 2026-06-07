#pragma once

#include "spareribs/core/simulation_pack.hpp"
#include "spareribs/internal/div_ceil.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>

namespace spareribs {

namespace detail::spike_tracker {

template <std::floating_point F>
__global__ void update_kernel(int* count, F const* future, F const* past, F threshold,
                              std::size_t length) {
    assert(threadIdx.y == 0u and threadIdx.z == 0u);
    unsigned int const g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < length) {
        if (future[g_idx] > threshold and past[g_idx] < threshold) {
            ++count[g_idx];
        }
    }
}

} // namespace detail::spike_tracker

template <std::floating_point F>
class SpikeTracker {
    int* count_;
    std::size_t length_;
    F threshold_;
    unsigned int block_threads_;

  public:
    SpikeTracker(std::size_t length, F threshold, unsigned int block_threads)
        : count_{nullptr}, length_{length}, threshold_{threshold}, block_threads_{block_threads} {
        cudaMalloc(&count_, length_ * sizeof(int));
    }
    ~SpikeTracker() { cudaFree(count_); }

    int* count() { return count_; }
    int const* count() const { return count_; }
    void update(SimulationPack<F> const& future, SimulationPack<F> const& past) const {
        assert(future.len() == length_);
        assert(past.len() == length_);
        unsigned int const grid_blocks = div_ceil(length_, block_threads_);
        detail::spike_tracker::update_kernel<<<grid_blocks, block_threads_>>>(
            count_, future.u(), past.u(), threshold_, length_);
    }
};

} // namespace spareribs
