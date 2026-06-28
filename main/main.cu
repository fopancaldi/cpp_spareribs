#include "spareribs/spareribs.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <vector>

template <typename>
struct debug_str;

int main() {
    using namespace spareribs;
    using my_float = float;

    constexpr std::size_t length = 10;

    SimulationPack<my_float> sp(length);
    std::vector<my_float> u_h(length, 0.f), v_h(length, 0.f);
    cudaMemcpy(sp.u(), u_h.data(), length * sizeof(my_float), cudaMemcpyHostToDevice);
    cudaMemcpy(sp.v(), v_h.data(), length * sizeof(my_float), cudaMemcpyHostToDevice);

    evolution_map::GaussianJump gj_em(my_float{1e2f}, length, 0, 8);
    for (int i = 0; i < 1; ++i) {
        gj_em.evolve_in_place(sp, 1.f);
    }

    cudaMemcpy(u_h.data(), sp.u(), length * sizeof(my_float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h.data(), sp.v(), length * sizeof(my_float), cudaMemcpyDeviceToHost);
    std::cout << std::scientific << std::setprecision(10);
    std::ranges::copy(u_h, std::ostream_iterator<my_float>(std::cout, "\t"));
    std::cout << '\n';
    std::ranges::copy(v_h, std::ostream_iterator<my_float>(std::cout, "\t"));
    std::cout << '\n';

    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
}
