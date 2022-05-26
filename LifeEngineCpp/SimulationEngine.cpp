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
        if (cp.build_threads) {
            SimulationEnginePartialMultiThread::kill_threads(dc);
            SimulationEnginePartialMultiThread::build_threads(dc, cp, sp);
        }
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else {cp.engine_paused = false;}
        if (cp.pause_processing_user_action) {cp.processing_user_actions = false;} else {cp.processing_user_actions = true;}
        if (cp.processing_user_actions) {process_user_action_pool();}
        if ((!cp.engine_paused || cp.engine_pass_tick) && (!cp.pause_button_pause || cp.pass_tick)) {
            //if ((!cp.engine_pause || cp.synchronise_simulation_tick) && !cp.engine_global_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                cp.pass_tick = false;
                cp.synchronise_simulation_tick = false;
                simulation_tick();
                if (sp.auto_food_drop_rate > 0) {random_food_drop();}
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
            SimulationEnginePartialMultiThread::build_threads(dc, cp, sp);
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

    if (dc.organisms.empty() && dc.to_place_organisms.empty()) {
        cp.organisms_extinct = true;
        if (sp.pause_on_total_extinction || sp.reset_on_total_extinction) {
            cp.engine_paused = true;
            return;
        }
    } else {
        cp.organisms_extinct = false;
    }

//    switch (cp.simulation_mode) {
//
//
//        case SimulationModes::CPU_Single_Threaded:
//        case SimulationModes::CPU_Partial_Multi_threaded:
//        case SimulationModes::CPU_Multi_Threaded:
//            if (dc.organisms.empty() && dc.to_place_organisms.empty()) {
//                cp.organisms_extinct = true;
//                if (sp.pause_on_total_extinction || sp.reset_on_total_extinction) {
//                    cp.engine_paused = true;
//                    return;
//                }
//            } else {
//                cp.organisms_extinct = false;
//            }
//            break;
//        case SimulationModes::GPU_CUDA_mode:
//            break;
//        case SimulationModes::OPENCL_MODE:
//            break;
//        case SimulationModes::GPUFORT_MODE:
//            break;
//    }

    switch (cp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            SimulationEngineSingleThread::single_threaded_tick(&dc, &sp, &mt);
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            SimulationEnginePartialMultiThread::partial_multi_thread_tick(&dc, &cp, &op, &sp, &mt);
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

//TODO refactor
void SimulationEngine::process_user_action_pool() {
    auto temp = std::vector<Organism*>{};
    for (auto & action: dc.user_actions_pool) {
        switch (action.type) {
            case ActionType::TryAddFood:
                if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::EmptyBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::FoodBlock;
                break;
            case ActionType::TryRemoveFood:
                if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::FoodBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = EmptyBlock;
                break;
            case ActionType::TryAddWall:
                if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::EmptyBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::WallBlock;
                break;
            case ActionType::TryRemoveWall:
                if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                if (action.x == 0 || action.y == 0 || action.x == dc.simulation_width-1 || action.y == dc.simulation_height-1) {continue;}
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::WallBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::EmptyBlock;
                break;
            case ActionType::TryAddOrganism:
                if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                break;
            case ActionType::TryKillOrganism: {
                    if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
                    if (dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::EmptyBlock ||
                        dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::WallBlock ||
                        dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::FoodBlock) { continue; }
                    Organism * organism_ptr = (dc.CPU_simulation_grid[action.x][action.y].organism);
                    //if (organism_ptr == nullptr) {continue;}
                    bool continue_flag = false;
                    for (auto & ptr: temp) {if (ptr == organism_ptr) {continue_flag=true; break;}}
                    if (continue_flag) {continue;}
                    temp.push_back(organism_ptr);
                    for (auto & block: organism_ptr->organism_anatomy->_organism_blocks) {
                        dc.CPU_simulation_grid
                        [organism_ptr->x + block.get_pos(organism_ptr->rotation).x]
                        [organism_ptr->y + block.get_pos(organism_ptr->rotation).y].type = BlockTypes::FoodBlock;
                    }
                    for (int i = 0; i < dc.organisms.size(); i++) {
                        if (dc.organisms[i] == organism_ptr) {
                            dc.organisms.erase(dc.organisms.begin() + i);
                            break;
                        }
                    }
                    delete organism_ptr;
                }
                break;
            case ActionType::TrySelectOrganism: {
                    if (check_if_out_of_bounds(&dc, action.x, action.y)) { continue; }
                    if (dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::EmptyBlock ||
                        dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::WallBlock ||
                        dc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::FoodBlock) { continue; }
                    dc.selected_organims = dc.CPU_simulation_grid[action.x][action.y].organism;
                }
                break;
        }
    }
    dc.user_actions_pool.clear();
}

void SimulationEngine::random_food_drop() {
    for (int i = 0; i < sp.auto_food_drop_rate; i++) {
        int x = std::uniform_int_distribution<int>(1, dc.simulation_width-2)(mt);
        int y = std::uniform_int_distribution<int>(1, dc.simulation_height-2)(mt);
        dc.user_actions_pool.push_back(Action{ActionType::TryAddFood, x, y});
    }
}

bool SimulationEngine::check_if_out_of_bounds(EngineDataContainer *dc, int x, int y) {
    return (x < 0 ||
            x > dc->simulation_width -1 ||
            y < 0 ||
            y > dc->simulation_height-1);
}