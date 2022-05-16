//
// Created by spaceeye on 07.05.2022.
//

#include "WindowCore.h"

int WindowCore::display_dialog_message(const std::string& message) {
    DescisionMessageBox msg{"Warning", QString::fromStdString(message), "OK", "Cancel", this};
    return msg.exec();
}

void WindowCore::display_message(const std::string &message) {
    QMessageBox msg;
    msg.setText(QString::fromStdString(message));
    msg.setWindowTitle("Warning");
    msg.exec();
}

template<typename T>
result_struct<T> WindowCore::try_convert_message_box_template(const std::string& message, QLineEdit *line_edit, T &fallback_value) {
    T result;
    if (boost::conversion::try_lexical_convert<T>(line_edit->text().toStdString(), result)) {
        return result_struct<T>{true, result};
    } else {
        display_message(message);
        line_edit->setText(QString::fromStdString(std::to_string(fallback_value)));
        return result_struct<T>{false, result};
    }
}

//==================== Toggle buttons ====================

void WindowCore::tb_pause_slot(bool paused) {
    cp.engine_global_pause = paused;
    parse_full_simulation_grid(cp.engine_global_pause);
}

void WindowCore::tb_stoprender_slot(bool stopped_render) {
    pause_image_construction = stopped_render;
    parse_full_simulation_grid(pause_image_construction);
    // calculating delta time is not needed when no image is being created.
    cp.calculate_simulation_tick_delta_time = !cp.calculate_simulation_tick_delta_time;
}

//==================== Buttons ====================

void WindowCore::b_clear_slot() {
    clear_world();
}

void WindowCore::b_reset_slot() {
    reset_world();
}

void WindowCore::b_resize_and_reset_slot() {
    resize_simulation_grid_flag = true;
    //resize_simulation_space();
}

void WindowCore::b_generate_random_walls_slot() {
    display_message("Not implemented");
}

void WindowCore::b_clear_all_walls_slot() {
    display_message("Not implemented");
}

void WindowCore::b_save_world_slot() {
    display_message("Not implemented");
}

void WindowCore::b_load_world_slot() {
    display_message("Not implemented");
}

void WindowCore::b_pass_one_tick_slot() {
    cp.engine_pass_tick = true;
    parse_full_simulation_grid(true);
}
void WindowCore::b_reset_view_slot() {
    reset_scale_view();
}

void WindowCore::b_kill_all_organisms_slot() {
    cp.engine_pause = true;
    wait_for_engine_to_pause();

    for (auto & organism: dc.organisms) {
        organism->lifetime = organism->max_lifetime*2;
    }
    for (auto & organism: dc.to_place_organisms) {
        organism->lifetime = organism->max_lifetime*2;
    }

    unpause_engine();
}

//==================== Line edits ====================

//There should be a better way of doing this, but I don't see one

void WindowCore::le_max_sps_slot() {
    int fallback = int(1/dc.simulation_interval);
    if (fallback < 0) {fallback = -1;}
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_sps, fallback);
    if (!result.is_valid) {return;}
    set_simulation_interval(result.result);
}

void WindowCore::le_max_fps_slot() {
    int fallback = int(1/window_interval);
    if (fallback < 0) {fallback = -1;}
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_fps, fallback);
    if (!result.is_valid) {return;}
    set_window_interval(result.result);
}

void WindowCore::le_num_threads_slot() {
    int fallback = int(cp.num_threads);
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_num_threads, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) { display_dialog_message("Number of threads cannot be less than 1."); return;}
    if (result.result > std::thread::hardware_concurrency()-1) {
        auto accept = display_dialog_message(
                "Warning, setting number of processes (" + std::to_string(result.result)
                + ") higher than the number of CPU cores (" +
                std::to_string(std::thread::hardware_concurrency()) +
                ") is not recommended, and will hurt the performance. To get the best result, try using less CPU threads than available CPU cores.");
        if (!accept) {return;}
    }
    set_simulation_num_threads(result.result);
}

//TODO cell size is not implemented
void WindowCore::le_cell_size_slot() {
    int fallback = cell_size;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_cell_size, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Size of cell cannot be less than 1."); return;}
    cell_size = result.result;
    display_message("Warning, changing cell size is not implemented");
}

//TODO not implemented
void WindowCore::le_simulation_width_slot() {
    int fallback = dc.simulation_width;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_simulation_width,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Simulation width cannot be less than 1."); return;}
    new_simulation_width = result.result;
}

void WindowCore::le_simulation_height_slot() {
    int fallback = dc.simulation_height;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_simulation_height,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Simulation height cannot be less than 1."); return;}
    new_simulation_height = result.result;
}

//I don't know how to do it better.
void WindowCore::le_food_production_probability_slot() {
    float fallback = sp.food_production_probability;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float",
                                                          _ui.le_food_production_probability, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0 || result.result > 1) {display_message("Input must be between 0 and 1."); return;}
    sp.food_production_probability = result.result;
}

void WindowCore::le_lifespan_multiplier_slot() {
    int fallback = sp.lifespan_multiplier;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_lifespan_multiplier,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.lifespan_multiplier = result.result;
}

//TODO set restraints
void WindowCore::le_look_range_slot() {
    int fallback = sp.look_range;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_look_range, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.look_range = result.result;
}

void WindowCore::le_auto_food_drop_rate_slot() {
    int fallback = sp.auto_food_drop_rate;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_auto_food_drop_rate,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.auto_food_drop_rate = result.result;
}

void WindowCore::le_extra_reproduction_cost_slot() {
    int fallback = sp.extra_reproduction_cost;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_extra_reproduction_cost,
                                                        fallback);
    if (!result.is_valid) {return;}
    sp.extra_reproduction_cost = result.result;
}

void WindowCore::le_global_mutation_rate_slot() {
    float fallback = sp.global_anatomy_mutation_rate;
    auto result = try_convert_message_box_template<float>("Inputted text is not an int", _ui.le_global_anatomy_mutation_rate,
                                                          fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.global_anatomy_mutation_rate = result.result;
}

void WindowCore::le_add_cell_slot() {
    int fallback = sp.add_cell;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_add, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.add_cell = result.result;
}

void WindowCore::le_change_cell_slot() {
    int fallback = sp.change_cell;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_change, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.change_cell = result.result;
}

void WindowCore::le_remove_cell_slot() {
    int fallback = sp.remove_cell;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_remove, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.remove_cell = result.result;
}

void WindowCore::le_min_reproducing_distance_slot() {
    int fallback = sp.min_reproducing_distance;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_min_reproduction_distance, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    if (result.result > sp.max_reproducing_distance) { display_message("Input cannot be more than max reproducing distance."); return;}
    sp.min_reproducing_distance = result.result;
}

void WindowCore::le_max_reproducing_distance_slot() {
    int fallback = sp.max_reproducing_distance;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_max_reproduction_distance, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    if (result.result < sp.min_reproducing_distance) { display_message("Input cannot be less than min reproducing distance."); return;}
    sp.max_reproducing_distance = result.result;
}

void WindowCore::le_max_organisms_slot() {
    int fallback = dc.max_organisms;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_max_organisms, fallback);
    if (!result.is_valid) {return;}
    dc.max_organisms = result.result;
}

void WindowCore::le_float_number_precision_slot() {
    int fallback = float_precision;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_float_number_precision, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    float_precision = result.result;
}

void WindowCore::le_killer_damage_amount_slot() {
    float fallback = sp.killer_damage_amount;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float", _ui.le_killer_damage_amount, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    sp.killer_damage_amount = result.result;
}
void WindowCore::le_produce_food_every_n_slot() {
    int fallback = sp.produce_food_every_n_life_ticks;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_produce_food_every_n_tick, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    sp.produce_food_every_n_life_ticks = result.result;
}

//==================== Radio buttond ====================

void WindowCore::rb_food_slot() {
    set_cursor_mode(CursorMode::Food_mode);
}

void WindowCore::rb_wall_slot() {
    set_cursor_mode(CursorMode::Wall_mode);
}

void WindowCore::rb_kill_slot() {
    set_cursor_mode(CursorMode::Kill_mode);
}

void WindowCore::rb_single_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Single_Threaded);
}

void WindowCore::rb_multi_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Multi_Threaded);
}

void WindowCore::rb_partial_multi_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Partial_Multi_threaded);
}

//TODO CUDA mode not implemented
void WindowCore::rb_cuda_slot() {
    return;
    set_simulation_mode(SimulationModes::GPU_CUDA_mode);
}

//==================== Check buttons ====================

void WindowCore::cb_stop_console_output_slot(bool state) {
    stop_console_output = state;
    if (state) {
        std::cout << "Console output stopped\n";
    } else {
        std::cout << "Console output resumed\n";
    }
}

void WindowCore::cb_synchronise_simulation_and_window_slot(bool state) {
    synchronise_simulation_and_window = state;
    cp.engine_pause = state;
}

void WindowCore::cb_use_evolved_anatomy_mutation_rate_slot(bool state) {
    sp.use_anatomy_evolved_mutation_rate = state;
    _ui.le_global_anatomy_mutation_rate->setDisabled(state);
}

void WindowCore::cb_use_evolved_brain_mutation_rate_slot(bool state) {
    sp.use_brain_evolved_mutation_rate = state;
    _ui.le_global_brain_mutation_rate->setDisabled(state);
}

void WindowCore::cb_reproduction_rotation_enabled_slot  (bool state) { sp.reproduction_rotation_enabled = state;}

void WindowCore::cb_on_touch_kill_slot                  (bool state) { sp.one_touch_kill = state;}

void WindowCore::cb_movers_can_produce_food_slot        (bool state) { sp.movers_can_produce_food = state;}

void WindowCore::cb_food_blocks_reproduction_slot       (bool state) { sp.food_blocks_reproduction = state;}

void WindowCore::cb_fill_window_slot                    (bool state) { fill_window = state;}

void WindowCore::cb_reset_on_total_extinction_slot      (bool state) { sp.reset_on_total_extinction = state;}

void WindowCore::cb_pause_on_total_extinction_slot      (bool state) { sp.pause_on_total_extinction = state;}

void WindowCore::cb_clear_walls_on_reset_slot           (bool state) { sp.clear_walls_on_reset = state;}

void WindowCore::cb_override_evolution_controls_slot    (bool state) { override_evolution_controls_slot = state;}

void WindowCore::cb_generate_random_walls_on_reset_slot (bool state) { sp.generate_random_walls_on_reset = state;}

void WindowCore::cb_runtime_rotation_enabled_slot       (bool state) { sp.runtime_rotation_enabled = state;}

void WindowCore::cb_fix_reproduction_distance_slot      (bool state) { sp.reproduction_distance_fixed = state;}