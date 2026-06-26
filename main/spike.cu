#include "spareribs/spareribs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <vector>

template <typename T>
    requires std::is_fundamental_v<T>
__global__ void print_kernel(T* ptr, std::size_t length, char c) {
    assert(threadIdx.x + threadIdx.y + threadIdx.z == 0);
    assert(blockIdx.x + blockIdx.y + blockIdx.z == 0);
    for (std::size_t i = 0; i < length; ++i) {
        printf("%c[%u]: %f\n", c, static_cast<unsigned int>(i), ptr[i]);
    }
}

template <std::floating_point F>
struct FakeFunction {
    __host__ __device__ spareribs::Vec2<F> operator()(spareribs::Vec2<F> vec) const {
        const auto [u, v] = vec;
        return {u, 0};
    }
};

int main() {
    namespace fs = std::filesystem;
    using namespace spareribs;
    using float_type = float;

    constexpr std::size_t simulations = 2, generator_seed = 0;
    constexpr unsigned int block_threads = 1 << 10, steps_until_check = 1000;
    constexpr float_type a = 1.3f, epsilon = .001f, T = .001f, step = .1f, threshold = .5f;
    constexpr float_type target_avg_spikes = 1;
    constexpr std::string_view outDirName = "spike_intervals";

    SimulationPack<float_type> sim_pack1(simulations), sim_pack2(simulations);
    curandGenerator_t u_generator;
    curandCreateGenerator(&u_generator, CURAND_RNG_PSEUDO_DEFAULT);
    /* curandSetPseudoRandomGeneratorSeed(u_generator, generator_seed);
    constexpr float_type u_0 = -a, v_0 = a * a * a / float_type{3} - a;
    generate_normal(u_generator, sim_pack1.u(), sim_pack1.len(), u_0,
                    T / std::sqrt(a * a - float_type{1}));
    const std::vector v_0_vec(sim_pack1.len(), v_0);
    cudaMemcpy(sim_pack1.v(), v_0_vec.data(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyHostToDevice); */
    const std::vector<float_type> u_0_vec(sim_pack1.len(), 0.3f);
    const std::vector<float_type> v_0_vec(sim_pack1.len(), 1);
    cudaMemcpy(sim_pack1.u(), u_0_vec.data(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyHostToDevice);
    cudaMemcpy(sim_pack1.v(), v_0_vec.data(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyHostToDevice);

    Rk4EvMap rk4_em(/* FhnNeuron{epsilon, a} */ FakeFunction<float_type>{}, float_type{});
    IndependentEvMap ind_em(rk4_em, block_threads, float_type{});
    /* GaussianJumpEvMap gj_em(T, sim_pack1.len(), generator_seed, block_threads);
    RibsCompositionEvMap evolution_map(ind_em, gj_em, float_type{}); */
    SpikeTracker<float_type> spike_tr(sim_pack1.len(), target_avg_spikes, threshold, block_threads);

    for (unsigned int steps_since_check = 0, i = 0; i < 10; ++steps_since_check, ++i) {
        SimulationPack<float_type> const& past = steps_since_check % 2 == 0 ? sim_pack1 : sim_pack2;
        SimulationPack<float_type>& future = steps_since_check % 2 == 0 ? sim_pack2 : sim_pack1;
        /* evolution_map */ ind_em.evolve_out_of_place(past.u(), past.v(), future.u(), future.v(),
                                                       step, sim_pack1.len());
        future.time() = i * step;
        spike_tr.update(future, past);

        print_kernel<<<1, 1>>>(future.u(), future.len(), 'u');
        print_kernel<<<1, 1>>>(future.v(), future.len(), 'v');
        print_kernel<<<1, 1>>>(spike_tr.spike_times_single_sim(0), 5, 's');

        if (steps_since_check == steps_until_check) {
            float_type const avg_spikes = spike_tr.average_spikes();
            if (avg_spikes >= target_avg_spikes) {
                break;
            } else {
                steps_since_check = 0;
                std::cout << "avg spikes / target_avg_spikes: " << avg_spikes << " / "
                          << target_avg_spikes << '\n';
            }
        }
    }

    fs::path const outDirPath = fs::current_path() / outDirName;
    fs::remove_all(outDirPath);
    fs::create_directory(outDirPath);
    std::vector<std::ofstream> ofstreams{};
    std::ranges::transform(std::views::iota(std::size_t{0}, simulations),
                           std::back_inserter(ofstreams), [outDirPath](std::size_t i) {
                               return std::ofstream(outDirPath / (std::to_string(i) + ".csv"));
                           });
    std::vector<std::size_t> spikes_each_sim(simulations);
    cudaMemcpy(spikes_each_sim.data(), spike_tr.written_elems(), simulations * sizeof(std::size_t),
               cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < spikes_each_sim.size(); ++i) {
        std::vector<std::size_t> spike_times_single_sim(spikes_each_sim[i]);
        cudaMemcpy(spike_times_single_sim.data(), spike_tr.spike_times_single_sim(i),
                   spike_times_single_sim.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost);
        std::ranges::copy(spike_times_single_sim,
                          std::ostream_iterator<float_type>(ofstreams[i], "\n"));
    }

    cudaDeviceSynchronize();
    cudaError_t const e = cudaGetLastError();
    if (e == cudaSuccess) {
        std::cout << "No error\n";
    } else {
        std::cout << "Got error: " << cudaGetErrorString(e) << '\n';
    }
}
