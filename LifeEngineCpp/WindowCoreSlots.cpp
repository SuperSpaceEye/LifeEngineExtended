//
// Created by spaceeye on 07.05.2022.
//

#include "WindowCore.h"

int WindowCore::display_dialog_message(const std::string& message) {
    if (disable_warnings) {return true;}
    DescisionMessageBox msg{"Warning", QString::fromStdString(message), "OK", "Cancel", this};
    return msg.exec();
}

void WindowCore::display_message(const std::string &message) {
    QMessageBox msg;
    msg.setText(QString::fromStdString(message));
    msg.setWindowTitle("Warning");
    msg.setWindowFlags(Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    msg.exec();
}

template<typename T>
result_struct<T> WindowCore::try_convert_message_box_template(const std::string& message, QLineEdit *line_edit, T &fallback_value) {
    T result;
    if (boost::conversion::try_lexical_convert<T>(line_edit->text().toStdString(), result)) {
        return result_struct<T>{true, result};
    } else {
        display_message(message);
        line_edit->setText(QString::fromStdString(to_str(fallback_value, 5)));
        return result_struct<T>{false, result};
    }
}

//==================== Toggle buttons ====================

void WindowCore::tb_pause_slot(bool paused) {
    cp.pause_button_pause = paused;
    parse_full_simulation_grid(cp.pause_button_pause);
    cp.tb_paused = paused;
}

void WindowCore::tb_stoprender_slot(bool stopped_render) {
    pause_image_construction = stopped_render;
    parse_full_simulation_grid(pause_image_construction);
    // calculating delta time is not needed when no image is being created.
    cp.calculate_simulation_tick_delta_time = !cp.calculate_simulation_tick_delta_time;
}

//==================== Buttons ====================

void WindowCore::b_clear_slot() {
    if (display_dialog_message("All organisms and simulation grid will be cleared.")) {
        clear_world();
    }
}

void WindowCore::b_reset_slot() {
    if (display_dialog_message("All organisms and simulation grid will be reset.")) {
        reset_world();
    }
}

void WindowCore::b_resize_and_reset_slot() {
    resize_simulation_grid_flag = true;
}

void WindowCore::b_generate_random_walls_slot() {
    if (!disable_warnings) {display_message("Not implemented");}
}

void WindowCore::b_clear_all_walls_slot() {
    cp.engine_pause = true;
    wait_for_engine_to_pause();

    for (int x = 1; x < dc.simulation_width-1; x++) {
        for (int y = 1; y < dc.simulation_height-1; y++) {
            if (dc.CPU_simulation_grid[x][y].type == BlockTypes::WallBlock) {
                dc.CPU_simulation_grid[x][y].type = BlockTypes::EmptyBlock;
            }
        }
    }

    unpause_engine();
}

void WindowCore::b_save_world_slot() {
    cp.engine_pause = true;
    cp.pause_processing_user_action = true;
    wait_for_engine_to_pause_force();
    wait_for_engine_to_pause_processing_user_actions();

    QString selected_filter;
    QFileDialog file_dialog{};
    file_dialog.setWindowFlags(Qt::CustomizeWindowHint | Qt::WindowTitleHint);

    auto file_name = file_dialog.getSaveFileName(this, tr("Save world"), "",
                                                 "Custom save type (*.tlfcpp);;JSON (*.json)", &selected_filter);
#ifndef __WIN32
    bool file_exists = std::filesystem::exists(file_name.toStdString());
#endif
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.tlfcpp)") {
        filetype = ".tlfcpp";
    } else if (selected_filter.toStdString() == "JSON (*.json)") {
        filetype = ".json";
    } else {
        unpause_engine();
        cp.pause_processing_user_action = false;
        return;
    }
    std::string full_path = file_name.toStdString();

#ifndef __WIN32
    if (!file_exists) {
        full_path = file_name.toStdString() + filetype;
    }
#endif

    if (filetype == ".tlfcpp") {
        std::ofstream out(full_path, std::ios::out | std::ios::binary);
        write_data(out);
        out.close();

    } else {
        write_json_data(full_path);
    }

    unpause_engine();
    cp.pause_processing_user_action = false;
}


//TODO add to simulation engine a function to call to place organism from to_place_organisms

void WindowCore::b_load_world_slot() {
    cp.engine_pause = true;
    cp.pause_processing_user_action = true;
    wait_for_engine_to_pause_force();
    wait_for_engine_to_pause_processing_user_actions();

    QString selected_filter;
    auto file_name = QFileDialog::getOpenFileName(this, tr("Load world"), "",
                                                  tr("Custom save type (*.tlfcpp);;JSON (*.json)"), &selected_filter);
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.tlfcpp)") {
        filetype = ".tlfcpp";
    } else if (selected_filter.toStdString() == "JSON (*.json)"){
        filetype = ".json";
    } else {
        unpause_engine();
        cp.pause_processing_user_action = false;
        return;
    }

    for (auto & organism: dc.organisms) {
        delete organism;
    }
    dc.organisms.clear();
    for (auto & organism: dc.to_place_organisms) {
        delete organism;
    }
    dc.to_place_organisms.clear();

    std::string full_path = file_name.toStdString();

    if (filetype == ".tlfcpp") {
        std::ifstream in(full_path, std::ios::in | std::ios::binary);
        read_data(in);
        in.close();

    } else if (filetype == ".json") {
        read_json_data(full_path);
    }

    unpause_engine();
    cp.pause_processing_user_action = false;
    initialize_gui_settings();
    update_table_values();
}

void WindowCore::b_pass_one_tick_slot() {
    cp.pass_tick = true;
    parse_full_simulation_grid(true);
}
void WindowCore::b_reset_view_slot() {
    reset_scale_view();
}

void WindowCore::b_kill_all_organisms_slot() {
    if (!display_dialog_message("All organisms will be killed.")) {return;}
    cp.engine_pause = true;
    wait_for_engine_to_pause_force();

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
    if (result.result > std::thread::hardware_concurrency()-1 && !disable_warnings) {
        auto accept = display_dialog_message(
                "Warning, setting number of processes (" + std::to_string(result.result)
                + ") higher than the number of CPU cores (" +
                std::to_string(std::thread::hardware_concurrency()) +
                ") is not recommended, and will hurt the performance. To get the best result, try using less CPU threads than available CPU cores.");
        if (!accept) {return;}
    }
    set_simulation_num_threads(result.result);
}

void WindowCore::le_cell_size_slot() {
    int fallback = cell_size;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_cell_size, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Size of cell cannot be less than 1."); return;}
    cell_size = result.result;
}

void WindowCore::le_simulation_width_slot() {
    int fallback = dc.simulation_width;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_simulation_width,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 10) {display_message("Simulation width cannot be less than 10."); return;}
    new_simulation_width = result.result;
}

void WindowCore::le_simulation_height_slot() {
    int fallback = dc.simulation_height;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_simulation_height,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 10) {display_message("Simulation height cannot be less than 10."); return;}
    new_simulation_height = result.result;
}

//I don't know how to do it better.
void WindowCore::le_food_production_probability_slot() {
    float fallback = sp.food_production_probability;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float",
                                                          _ui.le_food_production_probability, fallback);
    if (!result.is_valid) {return;}
    if (result.result <= 0 || result.result > 1) {display_message("Input must be between 0 and 1."); return;}
    sp.food_production_probability = result.result;
}

void WindowCore::le_lifespan_multiplier_slot() {
    int fallback = sp.lifespan_multiplier;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_lifespan_multiplier,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    sp.lifespan_multiplier = result.result;
}

void WindowCore::le_look_range_slot() {
    int fallback = sp.look_range;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_look_range, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    sp.look_range = result.result;
}

void WindowCore::le_auto_food_drop_rate_slot() {
    int fallback = sp.auto_produce_n_food;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_auto_produce_n_food,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.auto_produce_n_food = result.result;
}

void WindowCore::le_auto_produce_food_every_n_tick_slot() {
    int fallback = sp.auto_produce_food_every_n_ticks;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_auto_produce_food_every_n_tick,
                                                        fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    sp.auto_produce_food_every_n_ticks = result.result;
}

void WindowCore::le_extra_reproduction_cost_slot() {
    int fallback = sp.extra_reproduction_cost;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_extra_reproduction_cost,
                                                        fallback);
    if (!result.is_valid) {return;}
    sp.extra_reproduction_cost = result.result;
}

void WindowCore::le_global_anatomy_mutation_rate_slot() {
    float fallback = sp.global_anatomy_mutation_rate;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float", _ui.le_global_anatomy_mutation_rate,
                                                          fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    if (result.result > 1) {display_message("Input cannot be more than 1."); return;}
    sp.global_anatomy_mutation_rate = result.result;
}

void WindowCore::le_global_brain_mutation_rate_slot() {
    float fallback = sp.global_brain_mutation_rate;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float", _ui.le_global_brain_mutation_rate,
                                                          fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    if (result.result > 1) {display_message("Input cannot be more than 1."); return;}
    sp.global_brain_mutation_rate = result.result;
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
    if (result.result < 1) { display_message("Input cannot be less than 1."); return;}
    if (result.result > sp.max_reproducing_distance) { display_message("Input cannot be more than max reproducing distance."); return;}
    sp.min_reproducing_distance = result.result;
}

void WindowCore::le_max_reproducing_distance_slot() {
    int fallback = sp.max_reproducing_distance;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_max_reproduction_distance, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) { display_message("Input cannot be less than 1."); return;}
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

void WindowCore::le_anatomy_mutation_rate_delimiter_slot() {
    float fallback = sp.anatomy_mutation_rate_delimiter;
    auto result = try_convert_message_box_template<float>("Inputted text is not float", _ui.le_anatomy_mutation_rate_delimiter, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    if (result.result > 1) { display_message("Input cannot be more than 1."); return;}
    sp.anatomy_mutation_rate_delimiter = result.result;
}
void WindowCore::le_brain_mutation_rate_delimiter_slot() {
    float fallback = sp.brain_mutation_rate_delimiter;
    auto result = try_convert_message_box_template<float>("Inputted text is not float", _ui.le_brain_mutation_rate_delimiter, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) { display_message("Input cannot be less than 0."); return;}
    if (result.result > 1) { display_message("Input cannot be more than 1."); return;}
    sp.brain_mutation_rate_delimiter = result.result;
}

void WindowCore::le_font_size_slot() {
    auto _font = font();

    //font size could be set either by pixel_size or point_size. If it is set by one, the other will give -1.
    //so the program needs to understand which mode it is
    int font_size = 0;
    bool point_size_m;
    if (font().pixelSize() < 0) {
        font_size = font().pointSize();
        point_size_m = true;
    } else {
        font_size = font().pixelSize();
        point_size_m = false;
    }

    int fallback = font_size;
    auto result = try_convert_message_box_template<int>("Inputted text is not int", _ui.le_font_size, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}

    if (point_size_m) {
        _font.setPointSize(result.result);
    } else {
        _font.setPixelSize(result.result);
    }
    setFont(_font);
}

void WindowCore::le_max_move_range_slot() {
    int fallback = sp.max_move_range;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_max_move_range, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    if (result.result < sp.min_move_range) { display_message("Input cannot be less than min move distance."); return;}
    sp.max_move_range = result.result;
}

void WindowCore::le_min_move_range_slot() {
    int fallback = sp.min_move_range;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_min_move_range, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    if (result.result > sp.max_move_range) { display_message("Input cannot be more than max move distance"); return;}
    sp.min_move_range = result.result;
}

void WindowCore::le_move_range_delimiter_slot() {
    float fallback = sp.move_range_delimiter;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float", _ui.le_move_range_delimiter, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    if (result.result > 0) { display_message("Input cannot be more than 0"); return;}
    sp.move_range_delimiter = result.result;
}

void WindowCore::le_brush_size_slot() {
    int fallback = brush_size;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_brush_size, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    brush_size = result.result;
}

void WindowCore::le_update_info_every_n_milliseconds_slot() {
    int fallback = update_info_every_n_milliseconds;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_update_info_every_n_milliseconds, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    update_info_every_n_milliseconds = result.result;}

void WindowCore::le_menu_height_slot() {
    int fallback = _ui.menu_frame->frameSize().height();
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_menu_height, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 200) {display_message("Input cannot be less than 200."); return;}
    _ui.menu_frame->setFixedHeight(result.result);}

//==================== Radio button ====================

void WindowCore::rb_food_slot() {
    set_cursor_mode(CursorMode::ModifyFood);
}

void WindowCore::rb_wall_slot() {
    set_cursor_mode(CursorMode::ModifyWall);
}

void WindowCore::rb_kill_slot() {
    set_cursor_mode(CursorMode::KillOrganism);
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

void WindowCore::rb_cuda_slot() {
    set_simulation_mode(SimulationModes::GPU_CUDA_mode);
}

//==================== Check buttons ====================

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

void WindowCore::cb_fill_window_slot(bool state) {
    fill_window = state;
    if (!state) {
        le_simulation_width_slot();
        le_simulation_height_slot();
    }
}

void WindowCore::cb_use_nvidia_for_image_generation_slot(bool state) {
    if (!state) {
        use_cuda = false;
#if __CUDA_USED__
        cuda_creator.free();
#endif
        return;}

    auto result = cuda_is_available();
    if (!result) {
        _ui.cb_use_nvidia_for_image_generation->setChecked(false);
        use_cuda = false;
        if (!disable_warnings) {
            display_message("Warning, CUDA is not available on this device.");
        }
        return;
    }
    use_cuda = true;
}

void WindowCore::cb_reproduction_rotation_enabled_slot   (bool state) { sp.reproduction_rotation_enabled = state;}

void WindowCore::cb_on_touch_kill_slot                   (bool state) { sp.on_touch_kill = state;}

void WindowCore::cb_movers_can_produce_food_slot         (bool state) { sp.movers_can_produce_food = state;}

void WindowCore::cb_food_blocks_reproduction_slot        (bool state) { sp.food_blocks_reproduction = state;}

void WindowCore::cb_reset_on_total_extinction_slot       (bool state) { sp.reset_on_total_extinction = state;}

void WindowCore::cb_pause_on_total_extinction_slot       (bool state) { sp.pause_on_total_extinction = state;}

void WindowCore::cb_clear_walls_on_reset_slot            (bool state) { sp.clear_walls_on_reset = state;}

void WindowCore::cb_generate_random_walls_on_reset_slot  (bool state) { sp.generate_random_walls_on_reset = state;}

void WindowCore::cb_runtime_rotation_enabled_slot        (bool state) { sp.runtime_rotation_enabled = state;}

void WindowCore::cb_fix_reproduction_distance_slot       (bool state) { sp.reproduction_distance_fixed = state;}

void WindowCore::cb_disable_warnings_slot                (bool state) { disable_warnings = state;}

void WindowCore::cb_set_fixed_move_range_slot            (bool state) { sp.set_fixed_move_range = state;}

void WindowCore::cb_self_organism_blocks_block_sight_slot(bool state){ sp.organism_self_blocks_block_sight = state;}

void WindowCore::cb_failed_reproduction_eats_food_slot   (bool state) { sp.failed_reproduction_eats_food = state;}

void WindowCore::cb_wait_for_engine_to_stop_slot         (bool state) { wait_for_engine_to_stop = state;}

void WindowCore::cb_rotate_every_move_tick_slot          (bool state) { sp.rotate_every_move_tick = state;}

void WindowCore::cb_simplified_rendering_slot            (bool state) { simplified_rendering = state;}

void WindowCore::cb_apply_damage_directly_slot           (bool state) { sp.apply_damage_directly = state;}

void WindowCore::cb_multiply_food_production_prob_slot   (bool state) { sp.multiply_food_production_prob = state;}

void WindowCore::cb_simplified_food_production_slot      (bool state) { sp.simplified_food_production = state;}

void WindowCore::cb_stop_when_one_food_generated         (bool state) { sp.stop_when_one_food_generated = state;}

void WindowCore::cb_synchronise_info_with_window_slot    (bool state) { synchronise_info_with_window_update = state;}

void WindowCore::cb_eat_then_produce_slot                (bool state) { sp.eat_then_produce = state;}

//==================== Table ====================

void WindowCore::table_cell_changed_slot(int row, int col) {
    auto item = _ui.table_organism_block_parameters->item(row, col);
    float result;
    bool set_result = false;
    if (boost::conversion::try_lexical_convert<float>(item->text().toStdString(), result)) {
        set_result = true;
    } else {
        if (!disable_warnings) {display_message("Value should be float.");}
    }
    BlockParameters * type;
    switch (static_cast<BlocksNames>(row)) {
        case BlocksNames::MouthBlock:    type = &bp.MouthBlock;    break;
        case BlocksNames::ProducerBlock: type = &bp.ProducerBlock; break;
        case BlocksNames::MoverBlock:    type = &bp.MoverBlock;    break;
        case BlocksNames::KillerBlock:   type = &bp.KillerBlock;   break;
        case BlocksNames::ArmorBlock:    type = &bp.ArmorBlock;    break;
        case BlocksNames::EyeBlock:      type = &bp.EyeBlock;      break;
    }

    float * value;
    switch (static_cast<ParametersNames>(col)) {
        case ParametersNames::FoodCostModifier: value = &type->food_cost_modifier; break;
        case ParametersNames::LifePointAmount:  value = &type->life_point_amount;  break;
        case ParametersNames::ChanceWeight:     value = &type->chance_weight;      break;
    }

    if(set_result) {*value = result; return;}

    _ui.table_organism_block_parameters->item(row, col)->setText(QString::fromStdString(to_str(*value)));
    _ui.table_organism_block_parameters->update();
}