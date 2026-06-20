#include "spareribs/spareribs.hpp"

#include "spareribs/internal/curand_generate.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

int main() {
    using namespace spareribs;
    using float_type = float;

    constexpr std::size_t simulations = 1 << 10, generator_seed = 0;
    constexpr unsigned int block_threads = 1 << 10, steps_until_check = 1000;
    constexpr float_type a = 1.3f, epsilon = .001f, T = .001f, step = .1f, threshold = .5f;
    constexpr int target_avg_spikes = 1;
    constexpr std::string_view outFileName = "result.csv";

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

    Rk4EvMap rk4_em(FhnNeuron{epsilon, a}, float_type{});
    IndependentEvMap ind_em(rk4_em, block_threads, float_type{});
    GaussianJumpEvMap gj_em(T, sim_pack1.len(), generator_seed, block_threads);
    RibsCompositionEvMap evolution_map(ind_em, gj_em, float_type{});
    SpikeTracker<float_type> spike_tr(sim_pack1.len(), threshold, block_threads);

    for (unsigned int steps_since_check = 0;; ++steps_since_check) {
        SimulationPack<float_type> const& past = steps_since_check % 2 == 0 ? sim_pack1 : sim_pack2;
        SimulationPack<float_type>& future = steps_since_check % 2 == 0 ? sim_pack2 : sim_pack1;
        evolution_map.evolve_out_of_place(past.u(), past.v(), future.u(), future.v(), step,
                                          sim_pack1.len());
        spike_tr.update(future, past);

        if (steps_since_check == steps_until_check) {
            std::vector<int> spikes_h(sim_pack1.len());
            cudaMemcpy(spikes_h.data(), spike_tr.count(), sim_pack1.len() * sizeof(int),
                       cudaMemcpyDeviceToHost);
            int const avg_spikes = std::accumulate(spikes_h.cbegin(), spikes_h.cend(), 0) /
                                   static_cast<int>(sim_pack1.len());
            if (avg_spikes >= target_avg_spikes) {
                break;
            } else {
                steps_since_check = 0;
                std::cout << "avg spikes / target_avg_spikes: " << avg_spikes << " / "
                          << target_avg_spikes << '\n';
            }
        }
    }

    std::vector<float_type> u_final(sim_pack1.len()), v_final(sim_pack1.len());
    cudaMemcpy(u_final.data(), sim_pack1.u(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(v_final.data(), sim_pack1.v(), sim_pack1.len() * sizeof(float_type),
               cudaMemcpyDeviceToHost);
    std::vector<int> spikes_final(sim_pack1.len());
    cudaMemcpy(spikes_final.data(), spike_tr.count(), sim_pack1.len() * sizeof(int),
               cudaMemcpyDeviceToHost);
    std::ofstream ofs(outFileName.data());
    for (std::size_t i = 0; i < sim_pack1.len(); ++i) {
        ofs << spikes_final[i] << ',' << u_final[i] << ',' << v_final[i] << '\n';
    }

    cudaDeviceSynchronize();
    cudaError_t const e = cudaGetLastError();
    if (e == cudaSuccess) {
        std::cout << "No error\n";
    } else {
        std::cout << "Got error: " << cudaGetErrorString(e) << '\n';
    }
}
