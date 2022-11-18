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
                                   OCCParameters &occp) :
        edc(engine_data_container), ecp(engine_control_parameters), op(organism_block_parameters), sp(simulation_parameters),
        occp(occp) {
    boost::random_device rd;
    gen = lehmer64(rd());
    //TODO only do if enabled?
    init_auto_food_drop(edc.simulation_width, edc.simulation_height);
}

void SimulationEngine::threaded_mainloop() {
    auto point = std::chrono::high_resolution_clock::now();
    while (ecp.engine_working) {
        if (!edc.unlimited_simulation_fps && ecp.calculate_simulation_tick_delta_time) { point = std::chrono::high_resolution_clock::now();}
        if (ecp.stop_engine) {
            ecp.engine_working = false;
            ecp.engine_paused = true;
            ecp.stop_engine = false;
            return;
        }
        if (ecp.engine_pause || ecp.engine_global_pause) { ecp.engine_paused = true; } else { ecp.engine_paused = false;}
        process_user_action_pool();
        if ((!ecp.engine_paused || ecp.engine_pass_tick) && (!ecp.pause_button_pause || ecp.pass_tick)) {
            simulation_tick();
            ecp.engine_paused = false;
            ecp.engine_pass_tick = false;
            ecp.pass_tick = false;
            if (sp.auto_produce_n_food > 0) {random_food_drop();}
            if (edc.record_data) {
                edc.stc.tbuffer.record_recenter_to_imaginary_pos(sp.recenter_to_imaginary_pos);
                edc.stc.tbuffer.record_transaction();
            }
            if (edc.total_engine_ticks % ecp.update_info_every_n_tick == 0) {info.parse_info(&edc, &ecp);}
            if (ecp.execute_world_events && edc.total_engine_ticks % ecp.update_world_events_every_n_tick == 0) {
                world_events_controller.tick_events(edc.total_engine_ticks, ecp.pause_world_events);
            }
        }
        if (ecp.calculate_simulation_tick_delta_time) { edc.delta_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - point).count();}
        if (!edc.unlimited_simulation_fps) {std::this_thread::sleep_for(std::chrono::microseconds(int(edc.simulation_interval * 1000000 - edc.delta_time)));}
    }
}

//void SimulationEngine::change_mode() {
//    if (ecp.change_to_mode == ecp.simulation_mode) {
//        return;
//    }
//
//    //switches from
//    switch (ecp.simulation_mode) {
//        case SimulationModes::CPU_Single_Threaded:
//            break;
//        case SimulationModes::CPU_Partial_Multi_threaded:
//            break;
//        case SimulationModes::CPU_Multi_Threaded:
//            break;
//        case SimulationModes::GPU_CUDA_mode:
//            break;
//    }
//
//    //switches to
//    switch (ecp.change_to_mode) {
//        case SimulationModes::CPU_Single_Threaded:
//            break;
//        case SimulationModes::CPU_Partial_Multi_threaded:
//            break;
//        case SimulationModes::CPU_Multi_Threaded:
//            break;
//        case SimulationModes::GPU_CUDA_mode:
//            break;
//    }
//    ecp.simulation_mode = ecp.change_to_mode;
//}

void SimulationEngine::simulation_tick() {
    edc.engine_ticks_between_updates++;
    edc.total_engine_ticks++;

    ecp.organisms_extinct = edc.stc.num_alive_organisms == 0;

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
            ecp.tb_paused = true;
        }
        return;
    }

//    switch (ecp.simulation_mode){
//        case SimulationModes::CPU_Single_Threaded:
            SimulationEngineSingleThread::single_threaded_tick(&edc, &sp, &gen);
//            break;
//        case SimulationModes::CPU_Partial_Multi_threaded:
//            break;
//        case SimulationModes::CPU_Multi_Threaded:
//            break;
//        case SimulationModes::GPU_CUDA_mode:
//            break;
//    }
}

void SimulationEngine::process_user_action_pool() {
    ecp.do_not_use_user_actions_engine = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!ecp.do_not_use_user_actions_ui && !edc.ui_user_actions_pool.empty()) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::swap(edc.ui_user_actions_pool, edc.engine_user_actions_pool);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ecp.do_not_use_user_actions_engine = false;
    } else { ecp.do_not_use_user_actions_engine = false; return; }

    bool have_sorted = false;

    for (auto & action : edc.engine_user_actions_pool) {
        if (check_if_out_of_bounds(&edc, action.x, action.y)) {continue;}
        switch (action.type) {
            case ActionType::TryAddFood: {
                auto &type = edc.st_grid.get_type(action.x, action.y);
                if (type != BlockTypes::EmptyBlock) { continue; }
                //TODO
                auto & num = edc.st_grid.get_food_num(action.x, action.y);
                if (num + 1 > sp.max_food) { continue; }
                if (edc.record_data) { edc.stc.tbuffer.record_user_food_change(action.x, action.y, 1); }
                num+=1;
            }
                break;
            case ActionType::TryRemoveFood:
                try_remove_food(action.x, action.y);
                break;
            case ActionType::TryAddWall:
                set_wall(action);
                break;
            case ActionType::TryRemoveWall: {
                if (action.x == 0 || action.y == 0 || action.x == edc.simulation_width - 1 || action.y == edc.simulation_height - 1) {continue;}
                auto & type = edc.st_grid.get_type(action.x, action.y);
                if (type != BlockTypes::WallBlock) {continue;}
                type = BlockTypes::EmptyBlock;
                if (edc.record_data) { edc.stc.tbuffer.record_user_wall_change(action.x, action.y, false);}
            }
                break;
            case ActionType::TryAddOrganism: {
                bool continue_flag = false;
                edc.chosen_organism.init_values();

                continue_flag = action_check_if_space_for_organism_is_free(action, continue_flag);
                if (continue_flag) { continue; }
                if (!have_sorted) {OrganismsController::precise_sort_high_to_low_dead_organisms_positions(edc); have_sorted = true;}
                action_place_organism(action);
            }
                break;
            case ActionType::TryKillOrganism: {
                try_kill_organism(action.x, action.y);
                have_sorted = false;
            }
                break;
            case ActionType::TrySelectOrganism: {
                auto & type = edc.st_grid.get_type(action.x, action.y);
                if (type == BlockTypes::EmptyBlock || type == BlockTypes::WallBlock || type == BlockTypes::FoodBlock) { continue; }
                edc.selected_organism = OrganismsController::get_organism_by_index(edc.st_grid.get_organism_index(action.x, action.y), edc);
                goto endfor;
                }
        }
    }
    endfor:

    for (auto & action: edc.engine_user_actions_pool) {
        if (action.type == ActionType::TrySelectOrganism && edc.selected_organism != nullptr) {
            edc.chosen_organism.copy_organism(*edc.selected_organism);
            edc.selected_organism = nullptr;
            ecp.update_editor_organism = true;
            break;
        }
    }

    edc.engine_user_actions_pool.clear();
}

void SimulationEngine::action_place_organism(const Action &action) {
    auto * new_organism = OrganismsController::get_new_main_organism(edc);

    auto array_place = new_organism->vector_index;
    new_organism->copy_organism(edc.chosen_organism);
    new_organism->vector_index = array_place;

    if (array_place > edc.stc.last_alive_position) { edc.stc.last_alive_position = array_place; }

    new_organism->x = action.x;
    new_organism->y = action.y;

    if (edc.record_data) { edc.stc.tbuffer.record_user_new_organism(*new_organism);}

    for (auto &block: new_organism->anatomy._organism_blocks) {
        int x = block.get_pos(edc.chosen_organism.rotation).x + new_organism->x;
        int y = block.get_pos(edc.chosen_organism.rotation).y + new_organism->y;

        edc.st_grid.get_type(x, y)           = block.type;
        edc.st_grid.get_organism_index(x, y) = new_organism->vector_index;
        edc.st_grid.get_rotation(x, y)       = get_global_rotation(block.rotation, edc.chosen_organism.rotation);
    }
}

bool SimulationEngine::action_check_if_space_for_organism_is_free(const Action &action, bool continue_flag) {
    for (auto &block: edc.chosen_organism.anatomy._organism_blocks) {
        continue_flag = check_if_out_of_bounds(&edc,
                                               block.get_pos(edc.chosen_organism.rotation).x + action.x,
                                               block.get_pos(edc.chosen_organism.rotation).y + action.y);
        if (continue_flag) { break; }

        int x = block.get_pos(edc.chosen_organism.rotation).x + action.x;
        int y = block.get_pos(edc.chosen_organism.rotation).y + action.y;

        auto type = edc.st_grid.get_type(x, y);
        auto num = edc.st_grid.get_food_num(x, y);

        if (sp.food_blocks_reproduction) {
            if (type != BlockTypes::EmptyBlock || num > sp.food_threshold) {
                continue_flag = true;
                break;
            }
        } else {
            if (type != BlockTypes::EmptyBlock) {
                continue_flag = true;
                break;
            }
        }
    }
    return continue_flag;
}

void SimulationEngine::clear_walls() {
    for (int x = 1; x < edc.simulation_width - 1; x++) {
        for (int y = 1; y < edc.simulation_height - 1; y++) {
            auto & type = edc.st_grid.get_type(x, y);
            if (type == BlockTypes::WallBlock) {
                type = BlockTypes::EmptyBlock;
                if (edc.record_data) { edc.stc.tbuffer.record_user_wall_change(x, y, false);}
            }
        }
    }
}

void SimulationEngine::set_wall(const Action &action) {
    try_kill_organism(action.x, action.y);
    try_remove_food(action.x, action.y);
    edc.st_grid.get_type(action.x, action.y) = BlockTypes::WallBlock;
    if (edc.record_data) { edc.stc.tbuffer.record_user_wall_change(action.x, action.y, true);}
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
    auto & num = edc.st_grid.get_food_num(x, y);
    if (edc.record_data) {edc.stc.tbuffer.record_user_food_change(x, y, -num);}
    num = 0;
}

void SimulationEngine::try_kill_organism(int x, int y) {
    auto & type = edc.st_grid.get_type(x, y);
    if (type == BlockTypes::EmptyBlock || type == BlockTypes::WallBlock) { return; }
    Organism * organism_ptr = OrganismsController::get_organism_by_index(edc.st_grid.get_organism_index(x, y), edc);
    if (edc.record_data) {edc.stc.tbuffer.record_user_kill_organism(organism_ptr->vector_index);}
    for (auto & block: organism_ptr->anatomy._organism_blocks) {
        edc.st_grid.get_type(organism_ptr->x + block.get_pos(organism_ptr->rotation).x,
                             organism_ptr->y + block.get_pos(organism_ptr->rotation).y) = BlockTypes::EmptyBlock;
        //TODO
        edc.st_grid.get_food_num(organism_ptr->x + block.get_pos(organism_ptr->rotation).x,
                                 organism_ptr->y + block.get_pos(organism_ptr->rotation).y) += 1;
    }
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        if (&edc.stc.organisms[i] == organism_ptr) {
            edc.stc.organisms[i].kill_organism(edc);
            break;
        }
    }
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

    edc.base_organism.x = edc.simulation_width / 2;
    edc.base_organism.y = edc.simulation_height / 2;

    edc.chosen_organism.x = edc.simulation_width / 2;
    edc.chosen_organism.y = edc.simulation_height / 2;

    Organism * organism = OrganismsController::get_new_main_organism(edc);
    auto array_place = organism->vector_index;

    if (ecp.reset_with_editor_organism) {
        organism->copy_organism(edc.chosen_organism);
    } else {
        organism->copy_organism(edc.base_organism);
    }
    organism->vector_index = array_place;
    edc.stc.last_alive_position = organism->vector_index;

    if (edc.record_data) {
        edc.stc.tbuffer.record_new_organism(*organism);
        edc.stc.tbuffer.record_reset();
        edc.stc.tbuffer.record_transaction();}

    SimulationEngineSingleThread::place_organism(&edc, organism);

    if (ecp.execute_world_events) { stop_world_events(true); start_world_events();}

    ecp.engine_pass_tick = true;
}

void SimulationEngine::partial_clear_world() {
    clear_organisms();

    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            auto & type = edc.st_grid.get_type(x, y);

            if (type == BlockTypes::WallBlock) {
                if (!sp.clear_walls_on_reset) { continue;}
                if (edc.record_data) { edc.stc.tbuffer.record_user_wall_change(x, y, false);}
            }

            type = BlockTypes::EmptyBlock;

            edc.st_grid.get_food_num(x, y) = 0;
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
        edc.st_grid.get_type(x, 0) = BlockTypes::WallBlock;
        edc.st_grid.get_type(x, edc.simulation_height - 1) = BlockTypes::WallBlock;

        if (edc.record_data) {
            edc.stc.tbuffer.record_user_wall_change(x, 0, true);
            edc.stc.tbuffer.record_user_wall_change(x, edc.simulation_height - 1, true);}
    }

    for (int y = 0; y < edc.simulation_height; y++) {
        edc.st_grid.get_type(0, y) = BlockTypes::WallBlock;
        edc.st_grid.get_type(edc.simulation_width - 1, y) = BlockTypes::WallBlock;

        if (edc.record_data) {
            edc.stc.tbuffer.record_user_wall_change(0, y, true);
            edc.stc.tbuffer.record_user_wall_change(edc.simulation_width - 1, y, true);}
    }
}

void SimulationEngine::make_random_walls() {
    perlin.reseed(gen());

    auto temp = std::vector<Organism*>{};

    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            auto noise = perlin.octave2D_01((x * sp.perlin_x_modifier), (y * sp.perlin_y_modifier), sp.perlin_octaves, sp.perlin_persistence);

            if (noise >= sp.perlin_lower_bound && noise <= sp.perlin_upper_bound) {
                set_wall(Action{ActionType::TryAddWall, x, y});
            }
        }
    }
}

void SimulationEngine::reinit_organisms() {
    pause();

    for (auto & organism: edc.stc.organisms) {
        if (!organism.is_dead) {
            organism.init_values();
        }
    }
    for (auto & organism: edc.stc.child_organisms) {
        organism.init_values();
    }

    unpause();
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
bool SimulationEngine::wait_for_engine_to_pause() {
    while (!ecp.engine_paused) {}
    std::atomic_thread_fence(std::memory_order_release);
    do_not_unpause = false;
    return ecp.engine_paused;
}

void SimulationEngine::pause() {
    do_not_unpause = true;
    std::atomic_thread_fence(std::memory_order_release);
    ecp.engine_pause = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    wait_for_engine_to_pause();
}

void SimulationEngine::unpause() {
    if (do_not_unpause) { return;}
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ecp.engine_pause = false;
}

void SimulationEngine::parse_full_simulation_grid() {
    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            auto type = edc.st_grid.get_type(x, y);
            edc.simple_state_grid[x + y * edc.simulation_width].type = type;
            edc.simple_state_grid[x + y * edc.simulation_width].rotation = edc.st_grid.get_rotation(x, y);

            if (type == BlockTypes::EmptyBlock && edc.st_grid.get_food_num(x, y) > sp.food_threshold) {
                edc.simple_state_grid[x + y * edc.simulation_width].type = BlockTypes::FoodBlock;}
        }
    }
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
    stop_world_events_no_setting_reset();
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
    if (!ecp.execute_world_events) { return;}
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