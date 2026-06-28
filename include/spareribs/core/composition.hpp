#pragma once

#include "spareribs/core/concepts.hpp"
#include "spareribs/core/simulation_pack.hpp"

#include <concepts>

namespace spareribs::evolution_map {

template <typename TEvMap1, typename TEvMap2>
class RibsComposition {
    TEvMap1 ev_map_1_;
    TEvMap2 ev_map_2_;

  public:
    RibsComposition(TEvMap1 ev_map_1, TEvMap2 ev_map_2)
        : ev_map_1_{ev_map_1}, ev_map_2_{ev_map_2} {}

    template <concepts::gpu_float F>
        requires requires(TEvMap1 em1, TEvMap2 em2, SimulationPack<F>& p, F f) {
            { em1.evolve_in_place(p, f) } -> std::same_as<void>;
            { em2.evolve_in_place(p, f) } -> std::same_as<void>;
        }
    void evolve_in_place(SimulationPack<F>& pack, F step) {
        ev_map_1_.evolve_in_place(pack, step / F{2});
        ev_map_2_.evolve_in_place(pack, step);
        ev_map_1_.evolve_in_place(pack, step / F{2});
    }

    template <concepts::gpu_float F>
        requires requires(TEvMap1 em1, TEvMap2 em2, SimulationPack<F>& op,
                          SimulationPack<F> const& ip, F f) {
            { em1.evolve_in_place(op, f) } -> std::same_as<void>;
            { em1.evolve_out_of_place(op, ip, f) } -> std::same_as<void>;
            { em2.evolve_in_place(op, f) } -> std::same_as<void>;
        }
    void evolve_out_of_place(SimulationPack<F>& out_pack, SimulationPack<F> const& in_pack,
                             F step) {
        ev_map_1_.evolve_out_of_place(out_pack, in_pack, step / F{2});
        ev_map_2_.evolve_in_place(out_pack, step);
        ev_map_1_.evolve_in_place(out_pack, step / F{2});
    }
};

} // namespace spareribs::evolution_map
