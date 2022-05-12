//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters,
                                   std::mutex& mutex):
    mutex(mutex), dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters){

    mt = std::mt19937{rd()};
    dist = std::uniform_int_distribution<int>{0, 8};
}

//TODO refactor pausing/pass_tick/synchronise_tick
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
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else {cp.engine_paused = false;}
        process_user_action_pool();
        if (!cp.engine_paused || cp.engine_pass_tick) {
            if ((!cp.engine_pause || cp.synchronise_simulation_tick) && !cp.engine_global_pause) {
                cp.engine_paused = false;
                cp.engine_pass_tick = false;
                cp.synchronise_simulation_tick = false;
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
            single_threaded_tick(&dc, &sp, &mt);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            multi_threaded_tick();
            break;
        case SimulationModes::GPU_CUDA_mode:
            cuda_tick();
            break;
    }
}

void SimulationEngine::single_threaded_tick(EngineDataContainer * dc, SimulationParameters * sp, std::mt19937 * mt) {
    for (auto & organism: dc->organisms) {
        for (auto & block: organism.organism_anatomy->_organism_blocks) {
            dc->simulation_grid[organism.x + block.relative_x][organism.y + block.relative_y].type = block.organism_block.type;
        }
    }
//    for (auto & organism: dc->organisms) {
//
//    }
    auto to_erase = std::vector<int>{};

    for (auto & organism: dc->organisms) {produce_food(dc, sp, organism, *mt);}
    for (auto & organism: dc->organisms) {eat_food(dc, sp, organism);}
    for (int i = 0; i < dc->organisms.size(); i++) {tick_lifetime(dc, to_erase, dc->organisms[i], i);}
    for (int i = 0; i < to_erase.size(); ++i)      {erase_organisms(dc, to_erase, i);}
//    std::cout << dc->organisms[0].food_collected << "\n";
}

//Each producer will add one run of producing a food
void SimulationEngine::produce_food(EngineDataContainer * dc, SimulationParameters * sp, Organism & organism, std::mt19937 & mt) {
    for (int i = 0; i < organism.organism_anatomy->_producers; i++) {
        for (auto & pc: organism.organism_anatomy->_producing_space) {
            if (dc->simulation_grid[organism.x+pc.relative_x][organism.y+pc.relative_y].type == BlockTypes::EmptyBlock) {
                if (std::uniform_real_distribution<float>(0, 1)(mt) > sp->food_production_probability) {
                    dc->simulation_grid[organism.x+pc.relative_x][organism.y+pc.relative_y].type = BlockTypes::FoodBlock;
                    break;
                }
            }
        }
    }
}

void SimulationEngine::eat_food(EngineDataContainer * dc, SimulationParameters * sp, Organism & organism) {
    for (auto & pc: organism.organism_anatomy->_eating_space) {
        if (dc->simulation_grid[organism.x+pc.relative_x][organism.y+pc.relative_y].type == BlockTypes::FoodBlock) {
            dc->simulation_grid[organism.x+pc.relative_x][organism.y+pc.relative_y].type = BlockTypes::EmptyBlock;
            organism.food_collected++;
        }
    }
}

void SimulationEngine::tick_lifetime(EngineDataContainer *dc, std::vector<int>& to_erase, Organism &organism, int organism_pos) {
    organism.lifetime++;
    if (organism.lifetime > organism.max_lifetime || organism.damage > organism.life_points) {
        for (auto & block: organism.organism_anatomy->_organism_blocks) {
            dc->simulation_grid[organism.x+block.relative_x][organism.y+block.relative_y].type = BlockTypes::FoodBlock;
        }
        to_erase.push_back(organism_pos);
    }
}

void SimulationEngine::erase_organisms(EngineDataContainer *dc, std::vector<int> &to_erase, int i) {
    //when erasing organism vector will decrease, so we must account for that
    dc->organisms.erase(dc->organisms.begin() + to_erase[i] - i);
}


void SimulationEngine::tick_of_single_thread() {

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
