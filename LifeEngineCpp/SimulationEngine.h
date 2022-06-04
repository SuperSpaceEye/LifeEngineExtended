//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_LIFEENGINE_H
#define LANGUAGES_LIFEENGINE_H

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#include <boost/nondet_random.hpp>
#include <boost/random.hpp>

#include "GridBlocks/BaseGridBlock.h"
#include "Organism/CPU/Organism.h"
#include "BlockTypes.hpp"
#include "Containers/CPU/EngineControlContainer.h"
#include "Containers/CPU/EngineDataContainer.h"
#include "Containers/CPU/OrganismBlockParameters.h"
#include "Linspace.h"
#include "SimulationEngineModes/SimulationEnginePartialMultiThread.h"
#include "PRNGS/lehmer64.h"


//TODO move simulation grid translation to here
class SimulationEngine {
    EngineControlParameters& cp;
    EngineDataContainer& dc;
    OrganismBlockParameters& op;
    SimulationParameters& sp;

    std::mutex& mutex;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;

    void process_user_action_pool();

    void simulation_tick();
    void partial_multi_threaded_tick();
    void multi_threaded_tick();
    void cuda_tick();

    void change_mode();
    static bool check_if_out_of_bounds(EngineDataContainer *dc, int x, int y);

    void random_food_drop();

    //lehmer is like 2 times faster than mt19937
    lehmer64 gen;

    void try_kill_organism(int x, int y, std::vector<Organism*> & temp);
    void try_remove_food(int x, int y);

    void reset_world();
    void partial_clear_world();
    void clear_organisms();
    void make_walls();

public:
    SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                     OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters,
                     std::mutex& mutex);
    void threaded_mainloop();

};

#endif //LANGUAGES_LIFEENGINE_H
