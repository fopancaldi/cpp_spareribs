#pragma once

#include "spareribs/core/simulation_pack.hpp"
#include "spareribs/internal/div_ceil.hpp"

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace spareribs {

namespace detail::spike_tracker {

template <typename T>
__device__ T& ptr_at(T* ptr, std::size_t y_idx, std::size_t x_idx, std::size_t x_len) {
    return ptr[y_idx * x_len + x_idx];
}

template <std::floating_point F>
__global__ void update_kernel(F* spike_times, std::size_t* written_elems, F const* u_future,
                              F const* u_past, F current_time, F threshold,
                              std::size_t max_written_elems, std::size_t simulations) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    unsigned int const g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx < simulations) {
        if (u_future[g_idx] > threshold and u_past[g_idx] < threshold) {
            std::size_t const spikes_current_sim = written_elems[g_idx];
            assert(spikes_current_sim + 1 <= max_written_elems);
            ptr_at(spike_times, g_idx, spikes_current_sim, max_written_elems) = current_time;
            ++written_elems[g_idx];
        }
    }
}

template <std::integral I>
__host__ __device__ bool is_power_of_2(I i) {
    return (i & (i - 1)) == 0;
}

template <typename T>
    requires std::is_arithmetic_v<T> or std::is_floating_point_v<T>
__global__ void reduce_iter_kernel(T const* in, T* out, std::size_t in_length) {
    assert(threadIdx.y == 0);
    assert(threadIdx.z == 0);
    assert(blockIdx.y == 0);
    assert(blockIdx.z == 0);
    unsigned int const t_idx = threadIdx.x, b_idx = blockIdx.x, b_dim = blockDim.x,
                       g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ T shared[];
    assert(is_power_of_2(b_dim));

    shared[t_idx] = g_idx < in_length ? in[g_idx] : T{0};
    __syncthreads();

    for (unsigned int i = 1; i < b_dim; i *= 2) {
        if (t_idx % (2 * i) == 0) {
            shared[t_idx] += shared[t_idx + i];
        }
        __syncthreads();
    }

    out[b_idx] = shared[0];
}

template <typename T>
    requires std::is_arithmetic_v<T> or std::is_floating_point_v<T>
T* reduce(T const* values, std::size_t length, unsigned int block_threads) {
    T* current_values = nullptr;
    std::size_t current_length = length;
    while (current_length > 1) {
        T* reduced_values = nullptr;
        std::size_t const reduced_length = div_ceil(length, block_threads);
        cudaMalloc(&reduced_values, reduced_length * sizeof(T));
        T const* kernel_in = current_length < length ? current_values : values;
        reduce_iter_kernel<<<reduced_length, block_threads, block_threads * sizeof(T)>>>(
            kernel_in, reduced_values, current_length);
        if (current_length < length) {
            cudaFree(current_values);
        }
        current_values = reduced_values;
        current_length = reduced_length;
    }
    return current_values;
}

} // namespace detail::spike_tracker

template <std::floating_point F>
class SpikeTracker {
    F* spike_times_;
    std::size_t* written_elems_;
    std::size_t max_written_elems_;
    std::size_t simulations_;
    F threshold_;
    unsigned int block_threads_;

  public:
    SpikeTracker(std::size_t simulations, F expected_avg_spikes_, F threshold,
                 unsigned int block_threads)
        : spike_times_{nullptr}, written_elems_{nullptr},
          max_written_elems_{static_cast<std::size_t>(std::round(F{10} * expected_avg_spikes_))},
          simulations_{simulations}, threshold_{threshold}, block_threads_{block_threads} {
        cudaMalloc(&spike_times_, simulations_ * max_written_elems_ * sizeof(F));
        cudaMalloc(&written_elems_, simulations_ * sizeof(std::size_t));
        cudaMemset(written_elems_, 0x00, simulations_ * sizeof(std::size_t));
    }
    SpikeTracker(SpikeTracker<F> const&) = delete;
    SpikeTracker(SpikeTracker<F>&&) = default;
    SpikeTracker& operator=(SpikeTracker<F> const&) = delete;
    SpikeTracker& operator=(SpikeTracker<F>&&) = default;
    ~SpikeTracker() {
        cudaFree(&spike_times_);
        cudaFree(written_elems_);
    }

    std::size_t* written_elems() { return written_elems_; }
    F* spike_times_single_sim(std::size_t sim_idx) {
        assert(sim_idx < simulations_);
        return spike_times_ + sim_idx * max_written_elems_;
    }

    void update(SimulationPack<F> const& future, SimulationPack<F> const& past) const {
        assert(future.len() == simulations_);
        assert(past.len() == simulations_);
        unsigned int const grid_blocks = div_ceil(simulations_, block_threads_);
        detail::spike_tracker::update_kernel<<<grid_blocks, block_threads_>>>(
            spike_times_, written_elems_, future.u(), past.u(), future.time(), threshold_,
            max_written_elems_, simulations_);
    }
    std::size_t total_spikes() const {
        std::size_t result;
        std::size_t* const result_ptr =
            detail::spike_tracker::reduce(written_elems_, simulations_, block_threads_);
        cudaMemcpy(&result, result_ptr, sizeof(std::size_t), cudaMemcpyDeviceToHost);
        cudaFree(result_ptr);
        return result;
    }
    F average_spikes() const {
        std::size_t const total = total_spikes(), int_ratio = total / simulations_;
        return static_cast<F>(int_ratio) +
               static_cast<F>(total - int_ratio * simulations_) / static_cast<F>(simulations_);
    }
};

} // namespace spareribs
