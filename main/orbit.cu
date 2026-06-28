#include "spareribs/spareribs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    using namespace spareribs;
    using float_type = float;

    constexpr std::size_t simulations = 2, generator_seed = 0;
    constexpr unsigned int block_threads = 1 << 10, total_steps = 1000000;
    constexpr float_type a = 1.3f, epsilon = 0.01f, T = 0.01f, step_size = 0.1f;
    constexpr std::string_view outDirName = "orbits";
    assert(fs::current_path().filename() == "cpp_spareribs");

    SimulationPack<float_type> sim_pack(simulations);
    curandGenerator_t u_generator;
    curandCreateGenerator(&u_generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(u_generator, generator_seed);
    constexpr float_type u_0 = -a, v_0 = a * a * a / float_type{3} - a;
    generate_normal(u_generator, sim_pack.u(), sim_pack.len(), u_0,
                    T / std::sqrt(a * a - float_type{1}));
    const std::vector v_0_vec(sim_pack.len(), v_0);
    cudaMemcpy(sim_pack.v(), v_0_vec.data(), sim_pack.len() * sizeof(float_type),
               cudaMemcpyHostToDevice);

    OrbitTracker<float_type> orbit_tr(sim_pack.len(), total_steps);
    integrator::RungeKutta4 rk4_int(model::FhnNeuron{epsilon, a});
    evolution_map::Independent ind_em(rk4_int, block_threads);
    evolution_map::GaussianJump gj_em(T, sim_pack.len(), generator_seed, block_threads);
    evolution_map::RibsComposition evolution_map(ind_em, gj_em);

    for (unsigned int steps = 0; steps < total_steps; ++steps) {
        evolution_map.evolve_in_place(sim_pack, step_size);
        orbit_tr.fill_new(sim_pack);
        if (20 * steps % total_steps == 0) {
            std::cout << "steps / total_steps: " << steps << " / " << total_steps << '\n';
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
    for (unsigned int i = 0; i < total_steps; ++i) {
        std::vector<float_type> u(sim_pack.len()), v(sim_pack.len());
        cudaMemcpy(u.data(), orbit_tr.u_instants().at(i), u.size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), orbit_tr.v_instants().at(i), v.size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost);
        for (std::size_t j = 0; j < sim_pack.len(); ++j) {
            ofstreams[j] << u[j] << ',' << v[j] << '\n';
        }
    }

    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
}
