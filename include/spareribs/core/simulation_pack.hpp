#pragma once

#include <concepts>
#include <cstddef>
#include <utility>

namespace spareribs {

template <std::floating_point F>
class SimulationPack {
    F* u_;
    F* v_;
    std::size_t len_;
    F time_;

  public:
    SimulationPack() : u_{nullptr}, v_{nullptr}, len_{0}, time_{0} {}
    SimulationPack(std::size_t len) : u_{nullptr}, v_{nullptr}, len_{len}, time_{0} {
        cudaMalloc(&u_, len * sizeof(F));
        cudaMalloc(&v_, len * sizeof(F));
    }
    SimulationPack(SimulationPack<F> const& other) : SimulationPack(other.len_) {
        cudaMemcpy(u_, other.u_, len_ * sizeof(F), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_, other.v_, len_ * sizeof(F), cudaMemcpyDeviceToDevice);
    }
    SimulationPack(SimulationPack<F>&& other) : SimulationPack(other.len_) {
        u_ = other.u_;
        v_ = other.v_;
        other.u_ = nullptr;
        other.v_ = nullptr;
        other.len_ = 0;
    }
    SimulationPack& operator=(SimulationPack const& other) {
        if (this != &other) {
            SimulationPack tmp(other);
            swap(*this, tmp);
            return *this;
        }
    }
    SimulationPack& operator=(SimulationPack&& other) {
        if (this != &other) {
            swap(*this, other);
        }
        return *this;
    }
    ~SimulationPack() {
        cudaFree(u_);
        cudaFree(v_);
    }

    F* u() { return u_; }
    F* v() { return v_; }
    F const* u() const { return u_; }
    F const* v() const { return v_; }
    std::size_t len() const { return len_; }
    F& time() { return time_; }
    F time() const { return time_; }

    friend void swap(SimulationPack& lhs, SimulationPack& rhs) {
        std::swap(lhs.u_, rhs.u_);
        std::swap(lhs.v_, rhs.v_);
        std::swap(lhs.len_, rhs.len_);
    }
};

} // namespace spareribs
