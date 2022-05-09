//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, std::mutex& mutex):
    mutex(mutex), dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters){

    mt = std::mt19937{rd()};
    dist = std::uniform_int_distribution<int>{0, 8};
}

void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();
    while (cp.engine_working) {
//        if (cp.calculate_simulation_tick_delta_time) {point = std::chrono::high_resolution_clock::now();}
        //it works better without mutex... huh.
        //std::lock_guard<std::mutex> guard(mutex);
        if (cp.stop_engine) {
            kill_threads();
            cp.engine_working = false;
            cp.engine_paused = true;
            cp.stop_engine = false;
            return;
        }
        if (cp.change_simulation_mode) { change_mode(); }
        if (cp.build_threads) { build_threads(); }
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else { cp.engine_paused = false; }
        process_user_action_pool();
        if (!cp.engine_paused || cp.engine_pass_tick) {
            if (!cp.engine_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                simulation_tick();
//                if (cp.calculate_simulation_tick_delta_time) {dc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
//                if (!dc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(dc.simulation_interval * 1000000 - dc.delta_time)));}
            }
        }
    }
}

void SimulationEngine::change_mode() {
    if (cp.change_to_mode == cp.simulation_mode) {
        return;
    }

    switch (cp.change_to_mode) {
        case SimulationModes::CPU_Single_Threaded:
            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
                kill_threads();
            }
            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {

            }
            break;
        case SimulationModes::CPU_Multi_Threaded:
            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {

            }
            build_threads();
            break;
        case SimulationModes::GPU_CUDA_mode:
            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
                kill_threads();
            }
            break;
    }
    cp.simulation_mode = cp.change_to_mode;
}

void SimulationEngine::simulation_tick() {
    dc.engine_ticks++;

    switch (cp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            single_threaded_tick(&dc, &mt);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            multi_threaded_tick();
            break;
        case SimulationModes::GPU_CUDA_mode:
            cuda_tick();
            break;
    }
}

void SimulationEngine::single_threaded_tick(EngineDataContainer * dc, std::mt19937 * mt, int start_relative_x, int start_relative_y, int end_relative_x, int end_relative_y) {
    for (auto & organism: dc->organisms) {
        //std::cout << organism.organism_anatomy->_organism_blocks.size() << "\n";
        for (auto & block: organism.organism_anatomy->_organism_blocks) {
            //std::cout << block.relative_x << " " << block.relative_y << "\n";
            dc->simulation_grid[organism.x + block.relative_x][organism.y + block.relative_y].type = block.organism_block.type;
        }
    }
}

void SimulationEngine::multi_threaded_tick() {
    for (auto & thread :threads) {
        thread.work();
    }

    for (auto & thread: threads) {
        thread.finish();
    }
}

void SimulationEngine::cuda_tick() {

}

void SimulationEngine::kill_threads() {
    if (!threads.empty()) {
        for (auto & thread: threads) {
            thread.stop_work();
        }
        threads.clear();
    }
}

void SimulationEngine::build_threads() {
    kill_threads();
    threads.reserve(cp.num_threads);

    thread_points.clear();
    thread_points = Linspace<int>()(0, dc.simulation_width, cp.num_threads+1);

    for (int i = 0; i < cp.num_threads; i++) {
        threads.emplace_back(&dc, thread_points[i], 0, thread_points[i+1], dc.simulation_height);
    }
    cp.build_threads = false;
}