// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.03.2022.
//

#include "SimulationEngine.h"

SimulationEngine::SimulationEngine(EngineDataContainer &engine_data_container,
                                   EngineControlParameters &engine_control_parameters,
                                   OrganismBlockParameters &organism_block_parameters,
                                   SimulationParameters &simulation_parameters,
                                   RecordingData *recording_data, OCCParameters &occp) :
        edc(engine_data_container), ecp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters),
        recd(recording_data), occp(occp) {

    boost::random_device rd;
//    std::seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    gen = lehmer64(rd());
    //TODO only do if enabled?
    init_auto_food_drop(edc.simulation_width, edc.simulation_height);
}

//TODO it's kind of a mess right now
void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();

    while (ecp.engine_working) {
        if (!edc.unlimited_simulation_fps && ecp.calculate_simulation_tick_delta_time) { point = std::chrono::high_resolution_clock::now();}
        if (ecp.stop_engine) {
//            SimulationEnginePartialMultiThread::kill_threads(edc);
            ecp.engine_working = false;
            ecp.engine_paused = true;
            ecp.stop_engine = false;
            return;
        }
//        if (ecp.change_simulation_mode) { change_mode(); }
//        if (ecp.build_threads) {
////            SimulationEnginePartialMultiThread::build_threads(edc, ecp, sp);
//            ecp.build_threads = false;
//        }
        if (ecp.engine_pause || ecp.engine_global_pause) { ecp.engine_paused = true; } else { ecp.engine_paused = false;}
        process_user_action_pool();
        if ((!ecp.engine_paused || ecp.engine_pass_tick) && (!ecp.pause_button_pause || ecp.pass_tick)) {
            simulation_tick();
            ecp.engine_paused = false;
            ecp.engine_pass_tick = false;
            ecp.pass_tick = false;
            ecp.synchronise_simulation_tick = false;
            if (ecp.record_full_grid && edc.total_engine_ticks % ecp.parse_full_grid_every_n == 0) {parse_full_simulation_grid_to_buffer();}
            if (sp.auto_produce_n_food > 0) {random_food_drop();}
            if (edc.total_engine_ticks % ecp.update_info_every_n_tick == 0) {info.parse_info(&edc, &ecp);}
            if (ecp.execute_world_events && edc.total_engine_ticks % ecp.update_world_events_every_n_tick == 0) {
                world_events_controller.tick_events(edc.total_engine_ticks, ecp.pause_world_events);
            }
        }
        if (ecp.calculate_simulation_tick_delta_time) { edc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
        if (!edc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(edc.simulation_interval * 1000000 - edc.delta_time)));}
    }
}

void SimulationEngine::change_mode() {
    if (ecp.change_to_mode == ecp.simulation_mode) {
        return;
    }

    //switches from
    switch (ecp.simulation_mode) {
        case SimulationModes::CPU_Single_Threaded:
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
//            SimulationEnginePartialMultiThread::stop(edc, ecp, sp);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }

    //switches to
    switch (ecp.change_to_mode) {
        case SimulationModes::CPU_Single_Threaded:
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
//            SimulationEnginePartialMultiThread::init(edc, ecp, sp);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }
    ecp.simulation_mode = ecp.change_to_mode;
}

void SimulationEngine::simulation_tick() {
    edc.engine_ticks_between_updates++;
    edc.total_engine_ticks++;

//    switch (ecp.simulation_mode) {
//        case SimulationModes::CPU_Single_Threaded:
            if (edc.stc.num_alive_organisms == 0) {
                ecp.organisms_extinct = true;
            } else {
                ecp.organisms_extinct = false;
            }
//            break;
//        case SimulationModes::CPU_Partial_Multi_threaded:
//            ecp.organisms_extinct = true;
//            for (auto & pool: edc.organisms_pools) {
//                if (!pool.empty()) {
//                    ecp.organisms_extinct = false;
//                    break;
//                }
//            }
//            break;
//        default: break;
//    }

    if (ecp.organisms_extinct && (sp.pause_on_total_extinction || sp.reset_on_total_extinction)) {
        ecp.engine_paused = true;
        if (sp.reset_on_total_extinction) {
            reset_world();
            edc.auto_reset_counter++;
            if (sp.generate_random_walls_on_reset) {
                clear_walls();
                make_random_walls();
            }
        }
        if (sp.pause_on_total_extinction) {
            ecp.tb_paused = true ;
        }
        return;
    }

    switch (ecp.simulation_mode){
        case SimulationModes::CPU_Single_Threaded:
            SimulationEngineSingleThread::single_threaded_tick(&edc, &sp, &gen);
            break;
        case SimulationModes::CPU_Partial_Multi_threaded:
//            SimulationEnginePartialMultiThread::partial_multi_thread_tick(&edc, &ecp, &op, &sp, &gen);
            break;
        case SimulationModes::CPU_Multi_Threaded:
            break;
        case SimulationModes::GPU_CUDA_mode:
            break;
    }
}

//TODO refactor
void SimulationEngine::process_user_action_pool() {
    if (!edc.ui_user_actions_pool.empty() && !ecp.do_not_use_user_actions_ui) {
        ecp.do_not_use_user_actions_engine = true;
        edc.engine_user_actions_pool.resize(edc.ui_user_actions_pool.size());
        std::copy(edc.ui_user_actions_pool.begin(), edc.ui_user_actions_pool.end(), edc.engine_user_actions_pool.begin());
        edc.ui_user_actions_pool.clear();
        ecp.do_not_use_user_actions_engine = false;
    } else { return; }

    auto temp = std::vector<Organism*>{};
    for (auto & action : edc.engine_user_actions_pool) {
        if (check_if_out_of_bounds(&edc, action.x, action.y)) {continue;}
        switch (action.type) {
            case ActionType::TryAddFood:
                if (edc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::EmptyBlock) {continue;}
                edc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::FoodBlock;
                break;
            case ActionType::TryRemoveFood:
                try_remove_food(action.x, action.y);
                break;
            case ActionType::TryAddWall:
                set_wall(temp, action);
                break;
            case ActionType::TryRemoveWall:
                if (action.x == 0 || action.y == 0 || action.x == edc.simulation_width - 1 || action.y == edc.simulation_height - 1) {continue;}
                if (edc.CPU_simulation_grid[action.x][action.y].type != BlockTypes::WallBlock) {continue;}
                edc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::EmptyBlock;
                break;
            case ActionType::TryAddOrganism: {
                bool continue_flag = false;

                for (auto &block: edc.chosen_organism->anatomy._organism_blocks) {
                    continue_flag = check_if_out_of_bounds(&edc,
                                                           block.get_pos(edc.chosen_organism->rotation).x + action.x,
                                                           block.get_pos(edc.chosen_organism->rotation).y + action.y);
                    if (continue_flag) { break; }

                    int x = block.get_pos(edc.chosen_organism->rotation).x + action.x;
                    int y = block.get_pos(edc.chosen_organism->rotation).y + action.y;

                    auto & _block = edc.CPU_simulation_grid[x][y];

                    if (sp.food_blocks_reproduction) {
                        if (_block.type != BlockTypes::EmptyBlock) {
                            continue_flag = true;
                            break;
                        }
                    } else {
                        if (_block.type != BlockTypes::EmptyBlock && _block.type != BlockTypes::FoodBlock) {
                            continue_flag = true;
                            break;
                        }
                    }
                }

                if (continue_flag) { continue; }

                auto * new_organism = OrganismsController::get_new_main_organism(edc);

                auto array_place = new_organism->vector_index;
                *new_organism = Organism(edc.chosen_organism);
                new_organism->vector_index = array_place;

                if (array_place > edc.stc.last_alive_position) { edc.stc.last_alive_position = array_place; }

                new_organism->x = action.x;
                new_organism->y = action.y;

                for (auto &block: new_organism->anatomy._organism_blocks) {
                    int x = block.get_pos(edc.chosen_organism->rotation).x + new_organism->x;
                    int y = block.get_pos(edc.chosen_organism->rotation).y + new_organism->y;

                    edc.CPU_simulation_grid[x][y].type     = block.type;
                    edc.CPU_simulation_grid[x][y].organism_index = new_organism->vector_index;
                    edc.CPU_simulation_grid[x][y].rotation = get_global_rotation(block.rotation, edc.chosen_organism->rotation);
                }
            }
                break;
            case ActionType::TryKillOrganism: {
                try_kill_organism(action.x, action.y, temp);
            }
                break;
            case ActionType::TrySelectOrganism: {
                if (edc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::EmptyBlock ||
                    edc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::WallBlock  ||
                    edc.CPU_simulation_grid[action.x][action.y].type == BlockTypes::FoodBlock) { continue; }
                edc.selected_organism = OrganismsController::get_organism_by_index(edc.CPU_simulation_grid[action.x][action.y].organism_index, edc);
                goto endfor;
                }
        }
    }
    endfor:

    for (auto & action: edc.engine_user_actions_pool) {
        if (action.type == ActionType::TrySelectOrganism && edc.selected_organism != nullptr) {
            delete edc.chosen_organism;
            edc.chosen_organism = new Organism(edc.selected_organism);
            edc.selected_organism = nullptr;
            ecp.update_editor_organism = true;
            break;
        }
    }

    edc.engine_user_actions_pool.clear();
}

void SimulationEngine::clear_walls() {
    for (int x = 1; x < edc.simulation_width - 1; x++) {
        for (int y = 1; y < edc.simulation_height - 1; y++) {
            if (edc.CPU_simulation_grid[x][y].type == BlockTypes::WallBlock) {
                edc.CPU_simulation_grid[x][y].type = BlockTypes::EmptyBlock;
            }
        }
    }
}

void SimulationEngine::set_wall(std::vector<Organism *> &temp, const Action &action) {
    try_kill_organism(action.x, action.y, temp);
    try_remove_food(action.x, action.y);
    edc.CPU_simulation_grid[action.x][action.y].type = BlockTypes::WallBlock;
}

void SimulationEngine::random_food_drop() {
    if (sp.auto_produce_food_every_n_ticks <= 0) {return;}
    if (edc.total_engine_ticks % sp.auto_produce_food_every_n_ticks == 0) {
        for (int i = 0; i < sp.auto_produce_n_food; i++) {
            auto & vec = auto_food_drop_coordinates_shuffled[auto_food_drop_index % auto_food_drop_coordinates_shuffled.size()];
            auto_food_drop_index++;
            edc.ui_user_actions_pool.emplace_back(ActionType::TryAddFood, vec.x, vec.y);
        }
    }
}

void SimulationEngine::try_remove_food(int x, int y) {
    if (edc.CPU_simulation_grid[x][y].type != BlockTypes::FoodBlock) { return;}
    edc.CPU_simulation_grid[x][y].type = BlockTypes::EmptyBlock;
}

void SimulationEngine::try_kill_organism(int x, int y, std::vector<Organism*> & temp) {
    if (edc.CPU_simulation_grid[x][y].type == BlockTypes::EmptyBlock ||
        edc.CPU_simulation_grid[x][y].type == BlockTypes::WallBlock ||
        edc.CPU_simulation_grid[x][y].type == BlockTypes::FoodBlock) { return; }
    Organism * organism_ptr = OrganismsController::get_organism_by_index(edc.CPU_simulation_grid[x][y].organism_index, edc);
    bool continue_flag = false;
    for (auto & ptr: temp) {if (ptr == organism_ptr) {continue_flag=true; break;}}
    if (continue_flag) { return;}
    temp.push_back(organism_ptr);
    for (auto & block: organism_ptr->anatomy._organism_blocks) {
        edc.CPU_simulation_grid
        [organism_ptr->x + block.get_pos(organism_ptr->rotation).x]
        [organism_ptr->y + block.get_pos(organism_ptr->rotation).y].type = BlockTypes::FoodBlock;
    }
//    if (ecp.simulation_mode == SimulationModes::CPU_Single_Threaded) {
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        if (&edc.stc.organisms[i] == organism_ptr) {
            edc.stc.organisms[i].kill_organism(edc);
            break;
}
    }
//    } else if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
//        for (auto &pool: edc.organisms_pools) {
//            for (int i = 0; i < pool.size(); i++) {
//                if (pool[i] == organism_ptr) {
//                    pool.erase(pool.begin() + i);
//                    break;
//                }
//            }
//        }
//    }
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

    edc.base_organism->x = edc.simulation_width / 2;
    edc.base_organism->y = edc.simulation_height / 2;

    edc.chosen_organism->x = edc.simulation_width / 2;
    edc.chosen_organism->y = edc.simulation_height / 2;

    Organism * organism = OrganismsController::get_new_main_organism(edc);
    auto array_place = organism->vector_index;

    if (ecp.reset_with_editor_organism) {
        *organism = Organism(edc.chosen_organism);
    } else {
        *organism = Organism(edc.base_organism);
    }
    organism->vector_index = array_place;
    edc.stc.last_alive_position = organism->vector_index;

    SimulationEngineSingleThread::place_organism(&edc, organism);

    if (ecp.execute_world_events) { stop_world_events(true); start_world_events();}

    //Just in case
    ecp.engine_pass_tick = true;
    ecp.synchronise_simulation_tick = true;
}

void SimulationEngine::partial_clear_world() {
    clear_organisms();

    for (auto & column: edc.CPU_simulation_grid) {
        for (auto &block: column) {
            if (!sp.clear_walls_on_reset) {
                if (block.type == BlockTypes::WallBlock) { continue; }
            }
            block.type = BlockTypes::EmptyBlock;
        }
    }
    for (auto & block: edc.simple_state_grid) {
        if (!sp.clear_walls_on_reset) {
            if (block.type == BlockTypes::WallBlock) { continue; }
        }
        block.type = BlockTypes::EmptyBlock;
    }
    edc.total_engine_ticks = 0;
}

void SimulationEngine::clear_organisms() {
    edc.stc.organisms = std::vector<Organism>();
    edc.stc.child_organisms = std::vector<Organism>();
    edc.stc.dead_organisms_positions = std::vector<uint32_t>();
    edc.stc.free_child_organisms_positions = std::vector<uint32_t>();

    edc.stc.num_alive_organisms = 0;
    edc.stc.num_dead_organisms  = 0;
    edc.stc.last_alive_position = 0;
    edc.stc.dead_organisms_before_last_alive_position = 0;
}

void SimulationEngine::make_walls() {
    for (int x = 0; x < edc.simulation_width; x++) {
        edc.CPU_simulation_grid[x][0].type = BlockTypes::WallBlock;
        edc.CPU_simulation_grid[x][edc.simulation_height - 1].type = BlockTypes::WallBlock;
    }

    for (int y = 0; y < edc.simulation_height; y++) {
        edc.CPU_simulation_grid[0][y].type = BlockTypes::WallBlock;
        edc.CPU_simulation_grid[edc.simulation_width - 1][y].type = BlockTypes::WallBlock;
    }
}

void SimulationEngine::make_random_walls() {
    perlin.reseed(gen());

    auto temp = std::vector<Organism*>{};

    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            auto noise = perlin.octave2D_01((x * sp.perlin_x_modifier), (y * sp.perlin_y_modifier), sp.perlin_octaves, sp.perlin_persistence);

            if (noise >= sp.perlin_lower_bound && noise <= sp.perlin_upper_bound) {
                set_wall(temp, Action{ActionType::TryAddWall, x, y});
            }
        }
    }
}

void SimulationEngine::reinit_organisms() {
    ecp.engine_pause = true;
    while(!ecp.engine_paused) {}

//    switch (ecp.simulation_mode) {
//        case SimulationModes::CPU_Single_Threaded:
            for (auto & organism: edc.stc.organisms) {
                organism.init_values();
            }
//            break;
//        case SimulationModes::CPU_Partial_Multi_threaded:
//        case SimulationModes::CPU_Multi_Threaded:
//            for (auto & pool: edc.organisms_pools) {
//                for (auto & organism: pool) {
//                    organism->init_values();
//                }
//            }
//            break;
//        case SimulationModes::GPU_CUDA_mode:
//            break;
//    }

    ecp.engine_pause = false;
}

void SimulationEngine::init_auto_food_drop(int width, int height) {
    auto_food_drop_coordinates_shuffled.reserve((width-2)*(height-2));
    for (int x = 1; x < width-1; x++) {
        for (int y = 1; y < height-1; y++) {
            auto_food_drop_coordinates_shuffled.emplace_back(Vector2<int>{x, y});
        }
    }

    std::shuffle(auto_food_drop_coordinates_shuffled.begin(), auto_food_drop_coordinates_shuffled.end(), gen);
}

//Will always wait for engine to pause
bool SimulationEngine::wait_for_engine_to_pause_force() {
    while (!ecp.engine_paused) {}
    return ecp.engine_paused;
}

void SimulationEngine::pause() {
    ecp.engine_pause = true;
    wait_for_engine_to_pause_force();
}

void SimulationEngine::unpause() {
    if (!ecp.synchronise_simulation_and_window) {
        ecp.engine_pause = false;
    }
}

void SimulationEngine::parse_full_simulation_grid() {
    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            edc.simple_state_grid[x + y * edc.simulation_width].type = edc.CPU_simulation_grid[x][y].type;
            edc.simple_state_grid[x + y * edc.simulation_width].rotation = edc.CPU_simulation_grid[x][y].rotation;
        }
    }
}

void SimulationEngine::parse_full_simulation_grid_to_buffer() {
    while (ecp.pause_buffer_filling) {}
    ecp.recording_full_grid = true;
    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            recd->second_simulation_grid_buffer[recd->buffer_pos][x + y * edc.simulation_width].type = edc.CPU_simulation_grid[x][y].type;
            recd->second_simulation_grid_buffer[recd->buffer_pos][x + y * edc.simulation_width].rotation = edc.CPU_simulation_grid[x][y].rotation;
        }
    }
    recd->buffer_pos++;
    recd->recorded_states++;
    if (recd->buffer_pos >= recd->buffer_size) {
        recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc.simulation_width, edc.simulation_height, recd->second_simulation_grid_buffer);
        recd->buffer_pos = 0;
    }
    ecp.recording_full_grid = false;
}

void SimulationEngine::update_info() {
    info.parse_info(&edc, &ecp);
}

const OrganismInfoContainer & SimulationEngine::get_info() {
    return info.get_info();
}

void SimulationEngine::reset_world_events(std::vector<BaseEventNode *> start_nodes,
                                          std::vector<char> repeating_branch,
                                          std::vector<BaseEventNode *> node_storage) {
    stop_world_events();
    world_events_controller.reset_events(std::move(start_nodes), std::move(repeating_branch), std::move(node_storage));
    unpause();
}

void SimulationEngine::start_world_events() {
    pause();
    sp_copy = SimulationParameters{sp};
    ecp.execute_world_events = true;
    ecp.pause_world_events = false;
    unpause();
}

void SimulationEngine::resume_world_events() {
    ecp.pause_world_events = true;
}

void SimulationEngine::pause_world_events() {
    ecp.pause_world_events = false;
}

void SimulationEngine::stop_world_events(bool no_resume) {
    pause();
    sp = SimulationParameters{sp_copy};
    ecp.execute_world_events = false;
    ecp.pause_world_events = false;
    world_events_controller.reset();
    if (!no_resume) {unpause();}
}

void SimulationEngine::stop_world_events_no_setting_reset() {
    pause();
    ecp.execute_world_events = false;
    ecp.pause_world_events = false;
    world_events_controller.reset();
    unpause();
}

void SimulationEngine::set_seed(uint64_t new_seed) {
    gen.set_seed(new_seed);
}