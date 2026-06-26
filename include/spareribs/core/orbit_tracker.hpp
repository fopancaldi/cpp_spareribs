#pragma once

#include "spareribs/core/simulation_pack.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <vector>

namespace spareribs {

template <std::floating_point F>
class OrbitTracker {
    std::vector<F*> u_instants_;
    std::vector<F*> v_instants_;
    std::size_t simulations_;
    std::size_t instants_filled_;

  public:
    OrbitTracker() : u_instants_{}, v_instants_{}, simulations_{0}, instants_filled_{0} {}
    OrbitTracker(std::size_t simulations, std::size_t instants_total)
        : u_instants_(instants_total, nullptr), v_instants_(instants_total, nullptr),
          simulations_{simulations}, instants_filled_{0} {
        for (auto r : {std::ref(u_instants_), std::ref(v_instants_)}) {
            for (F*& p : r.get()) {
                cudaMalloc(&p, simulations * sizeof(F));
            }
        }
    }
    OrbitTracker(OrbitTracker<F> const&) = delete;
    OrbitTracker(OrbitTracker<F>&&) = default;
    OrbitTracker& operator=(OrbitTracker<F> const&) = delete;
    OrbitTracker& operator=(OrbitTracker<F>&&) = default;
    ~OrbitTracker() {
        for (auto r : {std::ref(u_instants_), std::ref(v_instants_)}) {
            for (F*& p : r.get()) {
                cudaFree(p);
            }
        }
    }

    std::vector<F*>& u_instants() { return u_instants_; }
    std::vector<F*>& v_instants() { return v_instants_; }
    std::vector<F*> const& u_instants() const { return u_instants_; }
    std::vector<F*> const& v_instants() const { return v_instants_; }

    void fill_new(SimulationPack<F> const& pack) {
        assert(instants_filled_ < u_instants_.size());
        cudaMemcpy(u_instants_[instants_filled_], pack.u(), simulations_ * sizeof(F),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_instants_[instants_filled_], pack.v(), simulations_ * sizeof(F),
                   cudaMemcpyDeviceToDevice);
        ++instants_filled_;
    }
};

} // namespace spareribs
