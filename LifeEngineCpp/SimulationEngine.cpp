//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, std::mutex& mutex):
    mutex(mutex), dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters){}

// TODO maybe not important. the majority of cpu time is spent on lock, but for now it's just an increment, so idk.
void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();
    while (cp.engine_working) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!cp.engine_global_pause || cp.engine_pass_tick) {
            if (!cp.engine_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                if (cp.calculate_simulation_tick_delta_time) {point = std::chrono::high_resolution_clock::now();}
                simulation_tick();
                if (cp.calculate_simulation_tick_delta_time) {dc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
            } else {
                cp.engine_paused = true;
            }
        } else {
            cp.engine_paused = true;
        }
    }
}

void SimulationEngine::simulation_tick() {
    dc.engine_ticks++;

//    for (int relative_x = 0; relative_x < dc.simulation_width; relative_x++) {
//        for (int relative_y = 0; relative_y < dc.simulation_height; relative_y++) {
//            dc.simulation_grid[relative_x][relative_y].type = static_cast<BlockTypes>(std::abs(dc.engine_ticks%9));
//        }
//    }
}



