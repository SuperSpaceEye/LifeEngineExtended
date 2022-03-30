//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_LIFEENGINE_H
#define LANGUAGES_LIFEENGINE_H

#include <atomic>
#include "iostream"
#include "vector"
#include "GridBlocks/BaseGridBlock.h"
#include "Organisms/Organism.h"
#include "mutex"
#include "BlockTypes.h"

class SimulationEngine {
    int simulation_width;
    int simulation_height;

    int& total_ticks;
    bool& working;
    bool& pause;
    bool& paused;
    bool& engine_global_pause;
    bool& engine_pass_tick;
    std::mutex& mutex;
    std::vector<std::vector<BaseGridBlock>>& simulation_grid;

    std::vector<Organism> organisms;
    // adding/killing organisms, adding/deleting food/walls, etc.
    //std::vector<Action> user_actions_pool;

    void process_user_action_pool();


public:
    SimulationEngine(int simulation_width, int simulation_height, std::vector<std::vector<BaseGridBlock>>& simulation_grid,
                     int& total_ticks, bool& working, bool& pause, bool& paused, bool& engine_global_pause, bool& engine_pass_tick, std::mutex& mutex);
    void simulation_tick();
    void threaded_mainloop();
};


#endif //LANGUAGES_LIFEENGINE_H
