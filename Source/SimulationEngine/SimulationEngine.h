// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

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
#include <algorithm>

#ifndef __EMSCRIPTEN_COMPILATION__
#include <boost/nondet_random.hpp>
#include <boost/random.hpp>
#endif

#include "Stuff/external/PerlinNoise.hpp"

#include "UIWindows/OrganismEditor/OrganismEditor.h"
#include "UIWindows/WorldEvents/WorldEventsController.h"

#include "Containers/EngineControlParametersContainer.h"
#include "Containers/EngineDataContainer.h"
#include "Containers/OrganismBlockParameters.h"
#include "Containers/OrganismInfoContainer.h"
#include "Stuff/Linspace.h"
#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/structs/Vector2.h"
#include "PRNGS/lehmer64.h"
#include "GridStuff/BaseGridBlock.h"
#include "Organism/Organism.h"

#include "SimulationEngineModes/SimulationEngineSingleThread.h"
#include "OrganismsController.h"

class SimulationEngine {
    EngineControlParameters& ecp;
    EngineDataContainer& edc;
    OrganismBlockParameters& op;
    SimulationParameters& sp;
    OCCParameters &occp;

    WorldEventsController world_events_controller{};
    SimulationParameters sp_copy;

    uint32_t auto_food_drop_index = 0;
    std::vector<Vector2<int>> auto_food_drop_coordinates_shuffled{};

    siv::PerlinNoise perlin;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;

    bool do_not_unpause = false;

    void process_user_action_pool();

    void simulation_tick();

    static bool check_if_out_of_bounds(EngineDataContainer *dc, int x, int y);

    void random_food_drop();

    //lehmer is like 2 times faster than mt19937
    lehmer64 gen;

    void try_kill_organism(int x, int y);
    void try_remove_food(int x, int y);
public:
    OrganismInfoContainer info{};

    SimulationEngine(EngineDataContainer &engine_data_container, EngineControlParameters &engine_control_parameters,
                     OrganismBlockParameters &organism_block_parameters, SimulationParameters &simulation_parameters, OCCParameters &occp);
    void threaded_mainloop();

    void make_random_walls();

    void set_wall(const Action &action);
    void clear_walls();

    void reinit_organisms();

    void reset_world();
    void partial_clear_world();
    void clear_organisms();
    void make_walls();

    void init_auto_food_drop(int width, int height);

    void pause();
    void unpause();

    bool wait_for_engine_to_pause();
    void parse_full_simulation_grid();

    void update_info();
    const OrganismInfoContainer & get_info();

    void start_world_events();
    void resume_world_events();
    void pause_world_events();
    void stop_world_events(bool no_resume = true);
    void stop_world_events_no_setting_reset();

    void reset_world_events(std::vector<WorldEventNodes::BaseEventNode *> start_nodes,
                            std::vector<char> repeating_branch,
                            std::vector<WorldEventNodes::BaseEventNode *> node_storage);

    void set_seed(uint64_t new_seed);

    bool action_check_if_space_for_organism_is_free(const Action &action, bool continue_flag);

    void action_place_organism(const Action &action);
};

#endif //LANGUAGES_LIFEENGINE_H
