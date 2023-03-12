// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.05.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
#define THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H


#include <iostream>
#include <vector>

#include "Containers/EngineControlParametersContainer.h"
#include "Containers/EngineDataContainer.h"
#include "Containers/OrganismBlockParameters.h"
#include "Organism/Organism.h"
#include "Organism/Brain/Observation.h"
#include "GridStuff/BaseGridBlock.h"
#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/Linspace.h"

inline Rotation get_global_rotation(Rotation rotation1, Rotation rotation2) {
    uint_fast8_t new_int_rotation = static_cast<uint_fast8_t>(rotation1) + static_cast<uint_fast8_t>(rotation2);
    return static_cast<Rotation>(new_int_rotation%4);
}

class SimulationEngineSingleThread {
public:
    static void place_organism(EngineDataContainer &edc, Organism &organism, SimulationParameters &sp);

    static void produce_food    (EngineDataContainer &edc, SimulationParameters &sp, Organism &organism, lehmer64 &gen);

    static void produce_food_simplified(EngineDataContainer &edc, SimulationParameters &sp,
                                        Organism &organism, lehmer64 &gen, float multiplier);

    static void produce_food_complex(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism, lehmer64 &gen, float multiplier);

    static void eat_food        (EngineDataContainer &edc, SimulationParameters &sp, Organism &organism);

    static void tick_lifetime(EngineDataContainer &edc, Organism &organism, int &i,
                              SimulationParameters &sp);

    static void apply_damage    (EngineDataContainer & edc, SimulationParameters & sp, Organism &organism);

    static void reserve_observations(std::vector<std::vector<Observation>> &observations,
                                     std::vector<Organism> &organisms,
                                     EngineDataContainer &edc);

    static void get_observations(EngineDataContainer &edc, SimulationParameters &sp,
                                 Organism &organism,
                                 std::vector<std::vector<Observation>> &organism_observations);

    static void rotate_organism(EngineDataContainer &edc, Organism &organism, BrainDecision decision,
                                SimulationParameters &sp, bool &moved);

    static void move_organism(EngineDataContainer &edc, Organism &organism, BrainDecision decision,
                              SimulationParameters &sp, bool &moved);

    static void make_decision   (EngineDataContainer &edc, SimulationParameters &sp, Organism &organism, lehmer64 &gen);

    static void try_make_child(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism,
                               lehmer64 &gen);

    static void place_child(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism,
                            lehmer64 &gen);

    static inline bool check_if_out_of_bounds(EngineDataContainer &edc, int x, int y);

    static inline bool check_if_block_out_of_bounds(EngineDataContainer &edc, Organism &organism,
                                             BaseSerializedContainer &block, Rotation rotation);

    static void single_threaded_tick(EngineDataContainer &edc, SimulationParameters &sp, lehmer64 &gen);

    static bool path_is_clear(int x, int y, Rotation direction, int steps, int32_t allow_organism, EngineDataContainer &edc,
                              SimulationParameters &sp);

    static void child_pos_calculator(Organism &organism, const Rotation to_place, int distance, EngineDataContainer &edc);

    //min x, min y, max x, max y
    static std::array<int, 4> get_organism_dimensions(Organism &organism);

    static bool calculate_discrete_movement(EngineDataContainer &edc, Organism &organism, BrainDecision decision,
                                            const SimulationParameters &sp, int &new_x, int &new_y);

    static bool
    calculate_continuous_move(EngineDataContainer &edc, Organism &organism, const SimulationParameters &sp, int &new_x,
                              int &new_y);
};


#endif //THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
