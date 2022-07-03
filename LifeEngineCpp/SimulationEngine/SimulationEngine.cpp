//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                                   OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters):
    dc(engine_data_container), cp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters){

    boost::random_device rd;
//    std::seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    gen = lehmer64(rd());
}

//TODO refactor pausing/pass_tick/synchronise_tick
//TODO takes 4.644% to 1% of processing time
void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();

    dc.single_thread_to_erase.reserve(dc.minimum_fixed_capacity);
    dc.single_thread_organisms_observations.reserve(dc.minimum_fixed_capacity);

    while (cp.engine_working) {
        if (!dc.unlimited_simulation_fps && cp.calculate_simulation_tick_delta_time) {point = std::chrono::high_resolution_clock::now();}
        if (cp.stop_engine) {
            SimulationEnginePartialMultiThread::kill_threads(dc);
            cp.engine_working = false;
            cp.engine_paused = true;
            cp.stop_engine = false;
            return;
        }
        if (cp.change_simulation_mode) { change_mode(); }
        if (cp.build_threads) {
            SimulationEnginePartialMultiThread::kill_threads(dc);
            SimulationEnginePartialMultiThread::build_threads(dc, cp, sp);
            cp.build_threads = false;
        }
        if (cp.engine_pause || cp.engine_global_pause) { cp.engine_paused = true; } else {cp.engine_paused = false;}
        if (cp.pause_processing_user_action) {cp.processing_user_actions = false;} else {cp.processing_user_actions = true;}
        if (cp.processing_user_actions) {process_user_action_pool();}
        if ((!cp.engine_paused || cp.engine_pass_tick) && (!cp.pause_button_pause || cp.pass_tick)) {
            cp.engine_paused = false;
            cp.engine_pass_tick = false;
            cp.pass_tick = false;
            cp.synchronise_simulation_tick = false;
            simulation_tick();
            if (sp.auto_produce_n_food > 0) {random_food_drop();}
            if (cp.calculate_simulation_tick_delta_time) {dc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
            if (!dc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(dc.simulation_interval * 1000000 - dc.delta_time)));}
        }
    }
}

void SimulationEngine::change_mode() {
    if (cp.change_to_mode == cp.simulation_mode) {
        return;
    }

    //switches from
    switch (cp.simulation_mode) {
        case SimulationModes::CPU_Single_Threaded:
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            SimulationEnginePartialMultiThread::stop(dc, cp, sp);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }

    //switches to
    switch (cp.change_to_mode) {
        case SimulationModes::CPU_Single_Threaded:
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            SimulationEnginePartialMultiThread::init(dc, cp, sp);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }
    cp.simulation_mode = cp.change_to_mode;
}

void SimulationEngine::simulation_tick() {
    dc.engine_ticks++;
    dc.total_engine_ticks++;

    switch (cp.simulation_mode) {
        case SimulationModes::CPU_Single_Threaded:
            if (dc.organisms.empty() && dc.to_place_organisms.empty()) {
                cp.organisms_extinct = true;
            } else {
                cp.organisms_extinct = false;
            }
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            cp.organisms_extinct = true;
            for (auto & pool: dc.organisms_pools) {
                if (!pool.empty() || !dc.to_place_organisms.empty()) {
                    cp.organisms_extinct = false;
                    break;
                }
            }
            break;
        default: break;
    }

    if (cp.organisms_extinct && (sp.pause_on_total_extinction || sp.reset_on_total_extinction)) {
        cp.engine_paused = true;
        if (sp.reset_on_total_extinction) {
            reset_world();
            dc.auto_reset_counter++;
        }
        if (sp.pause_on_total_extinction) {
            cp.tb_paused = true ;
            cp.organisms_extinct = false;
        }
        if (sp.generate_random_walls_on_reset) {
            clear_walls();
            make_random_walls();
        }
        return;
    }

    switch (cp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            SimulationEngineSingleThread::single_threaded_tick(&dc, &sp, &gen);
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
            SimulationEnginePartialMultiThread::partial_multi_thread_tick(&dc, &cp, &op, &sp, &gen);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }
}

//TODO refactor
void SimulationEngine::process_user_action_pool() {
    auto temp = std::vector<Organism*>{};
    for (auto & action: dc.user_actions_pool) {
        if (check_if_out_of_bounds(&dc, action.x, action.y)) {continue;}
        switch (action.type) {
            case ActionType::TryAddFood:
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::EmptyBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::FoodBlock;
                break;
            case ActionType::TryRemoveFood:
                try_remove_food(action.x, action.y);
                break;
            case ActionType::TryAddWall:
                set_wall(temp, action);
                break;
            case ActionType::TryRemoveWall:
                if (action.x == 0 || action.y == 0 || action.x == dc.simulation_width-1 || action.y == dc.simulation_height-1) {continue;}
                if (dc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::WallBlock) {continue;}
                dc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::EmptyBlock;
                break;
            case ActionType::TryAddOrganism:
                break;
            case ActionType::TryKillOrganism: {
                try_kill_organism(action.x, action.y, temp);
            }
                break;
            case ActionType::TrySelectOrganism: {
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

void SimulationEngine::clear_walls() {
    for (int x = 1; x < dc.simulation_width-1; x++) {
        for (int y = 1; y < dc.simulation_height-1; y++) {
            if (dc.CPU_simulation_grid[x][y].type == BlockTypes::WallBlock) {
                dc.CPU_simulation_grid[x][y].type = BlockTypes::EmptyBlock;
            }
        }
    }
}

void SimulationEngine::set_wall(std::vector<Organism *> &temp, const Action &action) {
    try_kill_organism(action.x, action.y, temp);
    try_remove_food(action.x, action.y);
    dc.CPU_simulation_grid[action.x][action.y].type = WallBlock;
}

void SimulationEngine::random_food_drop() {
    if (sp.auto_produce_food_every_n_ticks <= 0) {return;}
    if (dc.total_engine_ticks % sp.auto_produce_food_every_n_ticks == 0) {
        for (int i = 0; i < sp.auto_produce_n_food; i++) {
            int x = std::uniform_int_distribution<int>(1, dc.simulation_width - 2)(gen);
            int y = std::uniform_int_distribution<int>(1, dc.simulation_height - 2)(gen);
            dc.user_actions_pool.push_back(Action{ActionType::TryAddFood, x, y});
        }
    }
}

void SimulationEngine::try_remove_food(int x, int y) {
    if (dc.CPU_simulation_grid[x][y].type != BlockTypes::FoodBlock) { return;}
    dc.CPU_simulation_grid[x][y].type = EmptyBlock;
}

void SimulationEngine::try_kill_organism(int x, int y, std::vector<Organism*> & temp) {
    if (dc.CPU_simulation_grid[x][y].type == BlockTypes::EmptyBlock ||
        dc.CPU_simulation_grid[x][y].type == BlockTypes::WallBlock ||
        dc.CPU_simulation_grid[x][y].type == BlockTypes::FoodBlock) { return; }
    Organism * organism_ptr = (dc.CPU_simulation_grid[x][y].organism);
    bool continue_flag = false;
    for (auto & ptr: temp) {if (ptr == organism_ptr) {continue_flag=true; break;}}
    if (continue_flag) { return;}
    temp.push_back(organism_ptr);
    for (auto & block: organism_ptr->organism_anatomy->_organism_blocks) {
        dc.CPU_simulation_grid
        [organism_ptr->x + block.get_pos(organism_ptr->rotation).x]
        [organism_ptr->y + block.get_pos(organism_ptr->rotation).y].type = BlockTypes::FoodBlock;
    }
    if (cp.simulation_mode == SimulationModes::CPU_Single_Threaded) {
        for (int i = 0; i < dc.organisms.size(); i++) {
            if (dc.organisms[i] == organism_ptr) {
                dc.organisms.erase(dc.organisms.begin() + i);
                break;
            }
        }
    } else if (cp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        for (auto &pool: dc.organisms_pools) {
            for (int i = 0; i < pool.size(); i++) {
                if (pool[i] == organism_ptr) {
                    pool.erase(pool.begin() + i);
                    break;
                }
            }
        }
    }
    delete organism_ptr;
}

bool SimulationEngine::check_if_out_of_bounds(EngineDataContainer *dc, int x, int y) {
    return (x < 0 ||
            x > dc->simulation_width -1 ||
            y < 0 ||
            y > dc->simulation_height-1);
}

void SimulationEngine::reset_world() {
    partial_clear_world();
    make_walls();

    dc.base_organism->x = dc.simulation_width / 2;
    dc.base_organism->y = dc.simulation_height / 2;

    dc.chosen_organism->x = dc.simulation_width / 2;
    dc.chosen_organism->y = dc.simulation_height / 2;

    if (cp.reset_with_chosen) {dc.to_place_organisms.push_back(new Organism(dc.chosen_organism));}
    else                      {dc.to_place_organisms.push_back(new Organism(dc.base_organism));}

    //Just in case
    cp.engine_pass_tick = true;
    cp.synchronise_simulation_tick = true;
}

void SimulationEngine::partial_clear_world() {
    clear_organisms();

    for (auto & column: dc.CPU_simulation_grid) {
        for (auto &block: column) {
            if (!sp.clear_walls_on_reset) {
                if (block.type == BlockTypes::WallBlock) { continue; }
            }
            block.type = BlockTypes::EmptyBlock;
        }
    }
    for (auto & block: dc.second_simulation_grid) {
        if (!sp.clear_walls_on_reset) {
            if (block.type == BlockTypes::WallBlock) { continue; }
        }
        block.type = BlockTypes::EmptyBlock;
    }
    dc.total_engine_ticks = 0;
}

void SimulationEngine::clear_organisms() {
    if (cp.simulation_mode == SimulationModes::CPU_Single_Threaded) {
        for (auto &organism: dc.organisms) { delete organism; }
        dc.organisms.clear();
    } else if (cp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        for (auto &pool: dc.organisms_pools) {
            for (auto &organism: pool) { delete organism; }
            pool.clear();
        }
    }
    for (auto & organism: dc.to_place_organisms) {delete organism;}
    dc.to_place_organisms.clear();
}

void SimulationEngine::make_walls() {
    for (int x = 0; x < dc.simulation_width; x++) {
        dc.CPU_simulation_grid[x][0].type = BlockTypes::WallBlock;
        dc.CPU_simulation_grid[x][dc.simulation_height - 1].type = BlockTypes::WallBlock;
    }

    for (int y = 0; y < dc.simulation_height; y++) {
        dc.CPU_simulation_grid[0][y].type = BlockTypes::WallBlock;
        dc.CPU_simulation_grid[dc.simulation_width - 1][y].type = BlockTypes::WallBlock;
    }
}

void SimulationEngine::make_random_walls() {
    perlin.reseed(gen());

    auto temp = std::vector<Organism*>{};

    for (int x = 0; x < dc.simulation_width; x++) {
        for (int y = 0; y < dc.simulation_height; y++) {
            auto noise = perlin.octave2D_01((x * sp.perlin_x_modifier), (y * sp.perlin_y_modifier), sp.perlin_octaves, sp.perlin_persistence);

            if (noise >= sp.perlin_lower_bound && noise <= sp.perlin_upper_bound) {
                set_wall(temp, Action{ActionType::TryAddWall, x, y});
            }
        }
    }

}