// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

#include <vector>
#include <deque>

#include "../../Stuff/Actions.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "../../GridBlocks/SingleThreadGridBlock.h"
#include "../../Organism/CPU/ObservationStuff.h"
#include "../../Organism/CPU/Organism.h"
#include "OCCLogicContainer.h"

struct EngineDataContainer {
    uint64_t delta_time = 0;
    // for calculating ticks/second
    uint32_t engine_ticks_between_updates = 0;
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
        int32_t num_dead_organisms  = 0;
        int32_t num_alive_organisms = 0;
        //position of the last alive organism in organism vector.
        int32_t last_alive_position = 0;
        //number of organisms that are located before last_alive_position
        uint32_t dead_organisms_before_last_alive_position = 0;
        //factor determining how many dead_organisms_before_last_alive_position can be before compress_organisms is called
        float max_dead_organisms_in_alive_section_factor = 2;
        float memory_allocation_strategy_modifier = 2;
        std::vector<Organism> organisms{};
        std::vector<uint32_t> dead_organisms_positions{};
        std::vector<uint32_t> temp_dead_organisms_positions{};
        std::vector<Organism> child_organisms{};
        std::vector<uint32_t> free_child_organisms_positions{};
        std::vector<int> observation_count{};
        std::vector<std::vector<Observation>> organisms_observations{};

        OCCLogicContainer occ_container{};
    };
    SingleThreadContainer stc{};

    std::vector<BaseGridBlock> simple_state_grid;

    Organism * base_organism = nullptr;
    Organism * chosen_organism = nullptr;

    int auto_reset_counter = 0;

    // adding/killing organisms, adding/deleting food/walls, etc.
    std::vector<Action> ui_user_actions_pool;
    std::vector<Action> engine_user_actions_pool;

    Organism * selected_organism = nullptr;
};


#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
