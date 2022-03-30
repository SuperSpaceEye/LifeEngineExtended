//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(int simulation_width, int simulation_height,
                                   std::vector<std::vector<BaseGridBlock>>& simulation_grid,
                                   int& total_ticks, bool& working, bool& pause, bool& paused, bool& engine_global_pause,
                                   bool& engine_pass_tick, std::mutex& mutex):
    simulation_width(simulation_width), simulation_height(simulation_height), simulation_grid(simulation_grid),
    total_ticks(total_ticks), working(working), pause(pause), paused(paused), engine_global_pause(engine_global_pause),
    engine_pass_tick(engine_pass_tick), mutex(mutex){}

// TODO maybe not important. the majority of cpu time is spent on lock, but for now it's just an increment, so idk.
void SimulationEngine::threaded_mainloop() {
    while (working) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!engine_global_pause || engine_pass_tick) {
            if (!pause) {
                paused = false;
                engine_pass_tick = false;
                simulation_tick();
            } else {
                paused = true;
            }
        } else {
            paused = true;
        }
    }
}

void SimulationEngine::simulation_tick() {
    total_ticks++;

    for (int x = 0; x < simulation_width; x++) {
        for (int y = 0; y < simulation_height; y++) {
            simulation_grid[x][y].type = static_cast<BlockTypes>(std::abs(total_ticks%9));
        }
    }
}



