//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"
#include "SimulationEngineModes/SimulationEngineSingleThread.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters,
                                   std::mutex& mutex):
    mutex(mutex), dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters){

    boost::random_device rd;
    std::seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    mt = boost::mt19937(sd);
}

//TODO refactor pausing/pass_tick/synchronise_tick
void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();
    while (cp.engine_working) {
//        if (cp.calculate_simulation_tick_delta_time) {point = std::chrono::high_resolution_clock::now();}
        //it works better without mutex... huh.
        //std::lock_guard<std::mutex> guard(mutex);
        if (cp.stop_engine) {
            //kill_threads();
            cp.engine_working = false;
            cp.engine_paused = true;
            cp.stop_engine = false;
            return;
        }
        if (cp.change_simulation_mode) { change_mode(); }
        //if (cp.build_threads) { build_threads(); }
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else {cp.engine_paused = false;}
        process_user_action_pool();
        if ((!cp.engine_paused || cp.engine_pass_tick) && (!cp.pause_button_pause || cp.pass_tick)) {
            //if ((!cp.engine_pause || cp.synchronise_simulation_tick) && !cp.engine_global_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                cp.pass_tick = false;
                cp.synchronise_simulation_tick = false;
                simulation_tick();
//                if (cp.calculate_simulation_tick_delta_time) {dc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
//                if (!dc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(dc.simulation_interval * 1000000 - dc.delta_time)));}
            //}
        }
    }
}

void SimulationEngine::change_mode() {
    if (cp.change_to_mode == cp.simulation_mode) {
        return;
    }

    switch (cp.change_to_mode) {
        case SimulationModes::CPU_Single_Threaded:
//            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
//                kill_threads();
//            }
//            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {
//
//            }
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
//            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {
//
//            }
            break;
        case SimulationModes::CPU_Multi_Threaded:
//            if (cp.simulation_mode == SimulationModes::GPU_CUDA_mode) {
//
//            }
//            build_threads();
            break;
        case SimulationModes::GPU_CUDA_mode:
//            if (cp.simulation_mode == SimulationModes::CPU_Multi_Threaded) {
//                kill_threads();
//            }
            break;
        case SimulationModes::OPENCL_MODE:
            break;
        case SimulationModes::GPUFORT_MODE:
            break;
    }
    cp.simulation_mode = cp.change_to_mode;
}

void SimulationEngine::simulation_tick() {
    dc.engine_ticks++;
    dc.total_engine_ticks++;

    switch (cp.simulation_mode) {


        case SimulationModes::CPU_Single_Threaded:
        case SimulationModes::CPU_Partial_Multi_threaded:
        case SimulationModes::CPU_Multi_Threaded:
            if (dc.organisms.empty() && dc.to_place_organisms.empty()) {
                cp.organisms_extinct = true;
                if (sp.pause_on_total_extinction || sp.reset_on_total_extinction) {
                    cp.engine_paused = true;
                    return;
                }
            } else {
                cp.organisms_extinct = false;
            }
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
        case SimulationModes::OPENCL_MODE:
            break;
        case SimulationModes::GPUFORT_MODE:
            break;
    }

    switch (cp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            SimulationEngineSingleThread::single_threaded_tick(&dc, &sp, &mt);
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            break;
        case SimulationModes::CPU_Multi_Threaded:
            multi_threaded_tick();
            break;
        case SimulationModes::GPU_CUDA_mode:
            cuda_tick();
            break;
        case SimulationModes::OPENCL_MODE:
            break;
        case SimulationModes::GPUFORT_MODE:
            break;
    }
}

void SimulationEngine::partial_multi_threaded_tick() {

}

void SimulationEngine::multi_threaded_tick() {
//    for (auto & thread :threads) {
//        thread.work();
//    }
//
//    for (auto & thread: threads) {
//        thread.finish();
//    }
}

void SimulationEngine::cuda_tick() {

}