// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

#include "../../Stuff/Actions.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "../../GridBlocks/AtomicGridBlock.h"
#include "../../Organism/CPU/ObservationStuff.h"


struct eager_worker_partial;
class Organism;
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
    uint32_t simulation_width = 600;
    uint32_t simulation_height = 600;
    float simulation_interval = 0.;
    bool unlimited_simulation_fps = true;

    std::vector<std::vector<AtomicGridBlock>> CPU_simulation_grid;
    std::vector<Organism*> organisms;
    std::vector<Organism*> to_place_organisms;

    std::vector<int> single_thread_to_erase{};
    std::vector<int> single_thread_observation_count{};
    std::vector<std::vector<Observation>> single_thread_organisms_observations{};

    std::vector<std::vector<Organism*>> organisms_pools;

    std::vector<BaseGridBlock> second_simulation_grid;

    std::vector<eager_worker_partial> threads;
    std::vector<std::vector<int>> threaded_to_erase;
    std::vector<std::vector<std::vector<Observation>>> pooled_organisms_observations;
    std::vector<std::vector<pool_changes_info>> sorted_organisms_by_x_position;
    std::vector<std::vector<pool_changes_info*>> pool_changes;

    Organism * base_organism = nullptr;
    Organism * chosen_organism = nullptr;

    int auto_reset_counter = 0;

    // adding/killing organisms, adding/deleting food/walls, etc.
    std::vector<Action> user_actions_pool;

    uint32_t minimum_fixed_capacity = 100'000;

    uint32_t multithread_change_every_n_ticks = 1;

    Organism * selected_organism = nullptr;
};

struct pool_changes_info {
    Organism * organism = nullptr;
    int position_in_old_pool = -1;
    int old_pool = 500;
    int new_pool = 500;
};


#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
