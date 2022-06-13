//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

//#include "../../Organism/CPU/Organism.h"
#include "../../Actions.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "../../GridBlocks/AtomicGridBlock.h"
#include "../../Organism/CPU/ObservationStuff.h"

struct eager_worker_partial;
class Organism;

struct EngineDataContainer {
    uint64_t delta_time = 0;
    // for calculating ticks/second
    uint32_t engine_ticks = 0;
    // for tracking total ticks since start/reset of simulation.
    uint32_t total_engine_ticks = 0;
    // if -1, then unlimited
    int32_t max_organisms = -1;
    // dimensions of the simulation
    uint16_t simulation_width = 600;
    uint16_t simulation_height = 600;
    float simulation_interval = 0.;
    bool unlimited_simulation_fps = true;

    std::vector<std::vector<AtomicGridBlock>> CPU_simulation_grid;
    std::vector<Organism*> organisms;
    std::vector<Organism*> to_place_organisms;

    std::vector<std::vector<Organism*>> organisms_pools;

    std::vector<BaseGridBlock> second_simulation_grid;

    std::vector<eager_worker_partial> threads;
    std::vector<std::vector<int>> threaded_to_erase;
    std::vector<std::vector<std::vector<Observation>>> pooled_organisms_observations;

    Organism * base_organism;
    Organism * chosen_organism;

    int auto_reset_counter = 0;

    // adding/killing organisms, adding/deleting food/walls, etc.
    std::vector<Action> user_actions_pool;

    Organism * selected_organims = nullptr;
};


#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
