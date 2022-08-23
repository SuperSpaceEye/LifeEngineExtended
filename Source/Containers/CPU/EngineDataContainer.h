// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

#include <vector>

#include "../../Stuff/Actions.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "../../GridBlocks/SingleThreadGridBlock.h"
#include "../../Organism/CPU/ObservationStuff.h"
#include "../../Organism/CPU/Organism.h"

struct eager_worker_partial;
//class Organism;
struct pool_changes_info;

struct EngineDataContainer {
    uint64_t delta_time = 0;
    // for calculating ticks/second
    uint32_t engine_ticks = 0;
    // for tracking total ticks since start/reset of simulation.
    uint32_t total_engine_ticks = 0;

    uint32_t loaded_engine_ticks = 0;
    // if -1, then unlimited
    int32_t max_organisms = -1;
    // dimensions of the simulation
    int32_t simulation_width = 200;
    int32_t simulation_height = 200;
    float simulation_interval = 0.;
    bool unlimited_simulation_fps = true;

    std::vector<std::vector<SingleThreadGridBlock>> CPU_simulation_grid;

    struct SingleThreadContainer {
        float max_dead_to_alive_organisms_factor = 5;
        uint32_t num_dead_organisms  = 0;
        uint32_t num_alive_organisms = 0;
        //TODO process organisms from last_alive_position -> 0 so that organisms that are going to die could reduce the last_alive_position as they die.
        int32_t last_alive_position = 0;
        std::vector<Organism> organisms{};
        //Should be in the reverse order of the position of dead organism in main grid
        std::vector<uint32_t> dead_organisms_positions{};
        std::vector<Organism> child_organisms{};
        std::vector<uint32_t> free_child_organisms_positions{};
        std::vector<int> observation_count{};
        std::vector<std::vector<Observation>> organisms_observations{};
    };
    SingleThreadContainer stc{};

    std::vector<std::vector<Organism*>> organisms_pools;

    std::vector<BaseGridBlock> simple_state_grid;

//    std::vector<eager_worker_partial> threads;
    std::vector<std::vector<int>> threaded_to_erase;
    std::vector<std::vector<std::vector<Observation>>> pooled_organisms_observations;
    std::vector<std::vector<pool_changes_info>> sorted_organisms_by_x_position;
    std::vector<std::vector<pool_changes_info*>> pool_changes;

    //TODO
    Organism * base_organism = nullptr;
    Organism * chosen_organism = nullptr;

    int auto_reset_counter = 0;

    // adding/killing organisms, adding/deleting food/walls, etc.
    std::vector<Action> user_actions_pool;

    Organism * selected_organism = nullptr;
};

struct pool_changes_info {
    Organism * organism = nullptr;
    int position_in_old_pool = -1;
    int old_pool = 500;
    int new_pool = 500;
};


#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
