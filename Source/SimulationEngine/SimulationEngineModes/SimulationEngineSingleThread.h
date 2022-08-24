// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.05.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
#define THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H


#include <iostream>
#include <vector>

#include "../../GridBlocks/BaseGridBlock.h"
#include "../../Organism/CPU/Organism.h"
#include "../../Stuff/BlockTypes.hpp"
#include "../../Containers/CPU/EngineControlParametersContainer.h"
#include "../../Containers/CPU/EngineDataContainer.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "../../Stuff/Linspace.h"
#include "../../Organism/CPU/ObservationStuff.h"

inline Rotation get_global_rotation(Rotation rotation1, Rotation rotation2) {
    uint_fast8_t new_int_rotation = static_cast<uint_fast8_t>(rotation1) + static_cast<uint_fast8_t>(rotation2);
    return static_cast<Rotation>(new_int_rotation%4);
}

class SimulationEngineSingleThread {
public:
    static void place_organism  (EngineDataContainer * dc, Organism * organism);

    static void produce_food    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, lehmer64 &gen);

    static void produce_food_simplified(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, lehmer64 &gen, float multiplier);

    static void produce_food_complex(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, lehmer64 &gen, float multiplier);

    static void eat_food        (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void tick_lifetime(EngineDataContainer *dc, Organism *organism);

    static void apply_damage    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void reserve_observations(std::vector<std::vector<Observation>> &observations,
                                     std::vector<Organism> &organisms,
                                     EngineDataContainer *dc);

    static void get_observations(EngineDataContainer *dc, SimulationParameters *sp,
                                 Organism *organism,
                                 std::vector<std::vector<Observation>> &organism_observations);

    static void rotate_organism(EngineDataContainer *dc, Organism *organism, BrainDecision decision,
                                SimulationParameters *sp);

    static void move_organism(EngineDataContainer *dc, Organism *organism, BrainDecision decision,
                              SimulationParameters *sp);

    static void make_decision   (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, lehmer64 *gen);

    static void try_make_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                               lehmer64 *gen);

    static void place_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                            lehmer64 *gen);

    static bool check_if_out_of_bounds(EngineDataContainer *dc, int x, int y);

    static bool check_if_block_out_of_bounds(EngineDataContainer *dc, Organism *organism,
                                             BaseSerializedContainer &block, Rotation rotation);

    static void single_threaded_tick(EngineDataContainer * dc,
                                     SimulationParameters * sp,
                                     lehmer64 *gen);

    static bool path_is_clear(int x, int y, Rotation direction, int steps, int32_t allow_organism, EngineDataContainer *dc,
                              SimulationParameters *sp);

    static void new_child_pos_calculator(Organism *organism, Rotation to_place, int distance, EngineDataContainer &edc);

    static void old_child_pos_calculator(Organism *organism, Rotation to_place, int distance, EngineDataContainer &edc);

    //min x, min y, max x, max y
    static std::array<int, 4> get_organism_dimensions(Organism * organism);

    static void place_block_on_grid(EngineDataContainer *dc, Organism *organism, SerializedOrganismBlockContainer &block);
};


#endif //THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
