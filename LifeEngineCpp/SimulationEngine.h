//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_LIFEENGINE_H
#define LANGUAGES_LIFEENGINE_H

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include "GridBlocks/BaseGridBlock.h"
#include "Organisms/Organism.h"
#include "BlockTypes.h"
#include "EngineControlContainer.h"
#include "EngineDataContainer.h"
#include "OrganismBlockParameters.h"

class SimulationEngine {

    EngineControlParameters& cp;
    EngineDataContainer& dc;
    OrganismBlockParameters& op;

    std::mutex& mutex;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;

    void process_user_action_pool();


    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist;

public:
    SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                     OrganismBlockParameters& organism_block_parameters, std::mutex& mutex);
    void simulation_tick();
    void threaded_mainloop();
};


#endif //LANGUAGES_LIFEENGINE_H
