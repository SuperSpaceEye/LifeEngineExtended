//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_LIFEENGINE_H
#define LANGUAGES_LIFEENGINE_H

#include <iostream>
#include <vector>
#include <mutex>
#include "GridBlocks/BaseGridBlock.h"
#include "Organisms/Organism.h"
#include "BlockTypes.h"
#include "EngineControlContainer.h"
#include "EngineDataContainer.h"

class SimulationEngine {

    EngineControlParameters& cp;
    EngineDataContainer& dc;

    std::mutex& mutex;

    void process_user_action_pool();

public:
    SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters, std::mutex& mutex);
    void simulation_tick();
    void threaded_mainloop();
};


#endif //LANGUAGES_LIFEENGINE_H
