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
#include <vector>

int main() {
    namespace fs = std::filesystem;
    using namespace spareribs;
    using float_type = float;

    constexpr std::size_t simulations = 2, generator_seed = 0;
    constexpr unsigned int block_threads = 1 << 10, steps_until_check = 10000000;
    constexpr float_type a = 1.3f, epsilon = 0.01f, T = 0.01f, step_size = 0.1f,
                         spike_threshold = 0.f, min_spike_delay = 0.f;
    constexpr float_type target_avg_spikes = 1000;
    constexpr std::string_view outDirName = "spike_intervals";

    SimulationPack<float_type> sim_pack1(simulations), sim_pack2(simulations);
    curandGenerator_t u_generator;
    curandCreateGenerator(&u_generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(u_generator, generator_seed);
    constexpr float_type u_0 = -a, v_0 = a * a * a / float_type{3} - a;
    generate_normal(u_generator, sim_pack1.u(), sim_pack1.len(), u_0,
                    T / std::sqrt(a * a - float_type{1}));
    const std::vector v_0_vec(sim_pack1.len(), v_0);
    cudaMemcpy(sim_pack1.v(), v_0_vec.data(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyHostToDevice);
    sim_pack1.time() = 0.f;
    sim_pack2.time() = 0.f;

    integrator::RungeKutta4 rk4_int(model::FhnNeuron{epsilon, a});
    evolution_map::Independent ind_em(rk4_int, block_threads);
    evolution_map::GaussianJump gj_em(T, sim_pack1.len(), generator_seed, block_threads);
    evolution_map::RibsComposition evolution_map(ind_em, gj_em);
    SpikeTracker<float_type> spike_tr(sim_pack1.len(), target_avg_spikes, spike_threshold,
                                      block_threads);

    for (unsigned int steps = 0;; ++steps) {
        SimulationPack<float_type> const& past = steps % 2 == 0 ? sim_pack1 : sim_pack2;
        SimulationPack<float_type>& future = steps % 2 == 0 ? sim_pack2 : sim_pack1;
        evolution_map.evolve_out_of_place(future, past, step_size);
        future.time() = steps * step_size;
        spike_tr.update(future, past);

        if ((steps % steps_until_check) == 0) {
            cudaDeviceSynchronize();
            float_type const avg_spikes = spike_tr.average_spikes();
            if (avg_spikes >= target_avg_spikes) {
                break;
            } else {
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
                               std::ofstream r(outDirPath / (std::to_string(i) + ".csv"));
                               r << std::scientific << std::setprecision(10);
                               return r;
                           });
    std::vector<std::size_t> spikes_each_sim(simulations);
    cudaMemcpy(spikes_each_sim.data(), spike_tr.written_elems(), simulations * sizeof(std::size_t),
               cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < spikes_each_sim.size(); ++i) {
        if (spikes_each_sim[i] > 0) {
            std::vector<float_type> spike_times(spikes_each_sim[i]);
            cudaMemcpy(spike_times.data(), spike_tr.spike_times_single_sim(i),
                       spike_times.size() * sizeof(float_type), cudaMemcpyDeviceToHost);
            ofstreams[i] << spike_times.front() << '\n';
            for (auto it = spike_times.cbegin() + 1, last_written_it = spike_times.cbegin();
                 it != spike_times.cend(); ++it) {
                if (*it - *last_written_it > min_spike_delay) {
                    ofstreams[i] << *it - *last_written_it << '\n';
                    last_written_it = it;
                }
            }
        }
    }

    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
}
