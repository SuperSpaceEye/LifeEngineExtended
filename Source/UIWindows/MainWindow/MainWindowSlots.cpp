// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 07.05.2022.
//

#include "MainWindow.h"

//==================== Toggle buttons ====================

void MainWindow::tb_pause_slot(bool state) {
    ecp.pause_button_pause = state;
    parse_full_simulation_grid(ecp.pause_button_pause);
    ecp.tb_paused = state;
}

void MainWindow::tb_stoprender_slot(bool state) {
    pause_grid_parsing = state;
    if (!really_stop_render) {
        parse_full_simulation_grid(pause_grid_parsing);
    }
}

void MainWindow::tb_open_statistics_slot(bool state) {
    if (state) {
       st.show();
    } else {
       st.close();
    }
}

void MainWindow::tb_open_organism_editor_slot(bool state) {
    if (state) {
        ee.show();
        QTimer::singleShot(100, [&]{
            ee.reset_scale_view();
            ee.resize_image();
            ee.create_image();
        });
    } else {
        ee.close();
    }
}

void MainWindow::tb_open_info_window_slot(bool state) {
    if (state) {
        iw.show();
    } else {
        iw.close();
    }
}

void MainWindow::tb_open_recorder_window_slot(bool state) {
#ifndef __NO_RECORDER__
    if (state) {
        rc.show();
    } else {
        rc.close();
    }
#endif
}

void MainWindow::tb_open_world_events_slot(bool state) {
    if (state) {
        we.show();
    } else {
        we.close();
    }
}

void MainWindow::tb_open_benchmarks_slot(bool state) {
    if (state) {
        bs.show();
    } else {
        bs.close();
    }
}

void MainWindow::tb_open_occ_parameters_slot(bool state) {
    if (state) {
        occpw.show();
    } else {
        occpw.close();
    }
}

//==================== Buttons ====================

void MainWindow::b_clear_slot() {
    if (display_dialog_message("All organisms and simulation grid will be cleared.", disable_warnings)) {
        bool flag = sp.clear_walls_on_reset;
        sp.clear_walls_on_reset = true;
        engine.pause();
        clear_world();
        engine.unpause();
        sp.clear_walls_on_reset = flag;
    }
}

void MainWindow::b_reset_slot() {
    if (display_dialog_message("All organisms and simulation grid will be reset.", disable_warnings)) {
        engine.pause();
        if (ecp.reset_with_editor_organism) {ee.load_chosen_organism();}
        engine.reset_world();
        engine.unpause();
    }
}

void MainWindow::b_resize_and_reset_slot() {
    resize_simulation_grid_flag = true;
}

void MainWindow::b_generate_random_walls_slot() {
    engine.pause();
    engine.make_random_walls();
    engine.unpause();
}

void MainWindow::b_clear_all_walls_slot() {
    engine.pause();
    engine.clear_walls();
    engine.unpause();
}

void MainWindow::b_save_world_slot() {
    engine.pause();
    ecp.engine_global_pause = true;
    engine.wait_for_engine_to_pause();

    QString selected_filter;
    QFileDialog file_dialog{};

    std::atomic_thread_fence(std::memory_order_release);
    auto file_name = file_dialog.getSaveFileName(this, tr("Save world"), "",
                                                 "Custom save type (*.lfew);;JSON (*.json)", &selected_filter);
    std::atomic_thread_fence(std::memory_order_release);
#ifndef __WIN32
    bool file_exists = std::filesystem::exists(file_name.toStdString());
#endif
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.lfew)") {
        filetype = ".lfew";
    } else if (selected_filter.toStdString() == "JSON (*.json)") {
        filetype = ".json";
    } else {
        ecp.engine_global_pause = false;
        engine.unpause();
        return;
    }
    std::string full_path = file_name.toStdString();

#ifndef __WIN32
    if (!file_exists) {
        full_path = file_name.toStdString() + filetype;
    }
#endif

    if (filetype == ".lfew") {
        std::ofstream out(full_path, std::ios::out | std::ios::binary);
        write_data(out);
        out.close();
    } else {
        if (!sp.use_occ) {
            auto info = engine.get_info();
            DataSavingFunctions::write_json_data(full_path, edc, sp, info.total_total_mutation_rate);
        } else {
            display_message("Worlds cannot be saved in json format with OCC enabled.");
        }
    }

    ecp.engine_global_pause = false;
    engine.unpause();
}

void MainWindow::b_load_world_slot() {
    bool flag = sp.reset_on_total_extinction;
    sp.reset_on_total_extinction = false;
    engine.pause();
    ecp.engine_global_pause = true;
    engine.wait_for_engine_to_pause();

    std::atomic_thread_fence(std::memory_order_release);
    QString selected_filter;
    auto file_name = QFileDialog::getOpenFileName(this, tr("Load world"), "",
                                                  tr("Custom save type (*.lfew);;JSON (*.json)"), &selected_filter);
    std::atomic_thread_fence(std::memory_order_release);
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.lfew)") {
        filetype = ".lfew";
    } else if (selected_filter.toStdString() == "JSON (*.json)"){
        filetype = ".json";
    } else {
        ecp.engine_global_pause = false;
        engine.unpause();
        return;
    }

    std::string full_path = file_name.toStdString();

    if (filetype == ".lfew") {
        std::ifstream in(full_path, std::ios::in | std::ios::binary);
        read_world_data(in);
        in.close();

    } else if (filetype == ".json") {
        sp.use_occ = false;
        read_json_world_data(full_path);
    }

    sp.reset_on_total_extinction = flag;
    ecp.engine_global_pause = false;
    engine.unpause();
    initialize_gui();
    update_table_values();

    occpw.reinit_gui(true);
}

void MainWindow::b_pass_one_tick_slot() {
    ecp.pass_tick = true;
    parse_full_simulation_grid(true);
}
void MainWindow::b_reset_view_slot() {
    reset_scale_view();
}

void MainWindow::b_kill_all_organisms_slot() {
    if (!display_dialog_message("All organisms will be killed.", disable_warnings)) {return;}
    engine.pause();

    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        edc.stc.organisms[i].lifetime = edc.stc.organisms[i].max_lifetime + 5;
    }

    engine.unpause();
}

void MainWindow::b_update_textures_slot() { update_textures = true; }

//==================== Line edits ====================

void MainWindow::le_max_sps_slot() {
    int fallback = int(1 / edc.simulation_interval);
    if (fallback < 0) {fallback = -1;}
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", ui.le_sps, fallback);
    if (!result.is_valid) {return;}
    set_simulation_interval(result.result);
}

void MainWindow::le_max_fps_slot() {
    int fallback = int(1 / image_creation_interval);
    if (fallback < 0) {fallback = -1;}
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", ui.le_fps, fallback);
    if (!result.is_valid) {return;}
    set_image_creator_interval(result.result);
}

void MainWindow::le_num_threads_slot() {
    int fallback = int(ecp.num_threads);
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", ui.le_num_threads, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) { display_dialog_message("Number of threads cannot be less than 1.", disable_warnings); return;}
    if (result.result > std::thread::hardware_concurrency()-1 && !disable_warnings) {
        auto accept = display_dialog_message(
                "Warning, setting number of processes (" + std::to_string(result.result)
                + ") higher than the number of CPU cores (" +
                std::to_string(std::thread::hardware_concurrency()) +
                ") is not recommended, and will hurt the performance. To get the best result, try using less CPU threads than available CPU cores.",
                disable_warnings);
        if (!accept) {return;}
    }
    set_simulation_num_threads(result.result);
}

void MainWindow::le_cell_size_slot() {
    le_slot_lower_bound<int>(starting_cell_size_on_resize, starting_cell_size_on_resize, "int",
                             ui.le_cell_size, 1, "1");
}

void MainWindow::le_simulation_width_slot() {
    le_slot_lower_bound<int32_t>(edc.simulation_width, new_simulation_width, "int",
                                 ui.le_simulation_width, 10, "10");
}

void MainWindow::le_simulation_height_slot() {
    le_slot_lower_bound<int32_t>(edc.simulation_height, new_simulation_height, "int",
                                 ui.le_simulation_height, 10, "10");
}

void MainWindow::le_food_production_probability_slot() {
    le_slot_lower_upper_bound<float>(sp.food_production_probability, sp.food_production_probability, "float",
                                     ui.le_food_production_probability, 0, "0", 1, "1");
}

void MainWindow::le_lifespan_multiplier_slot() {
    le_slot_lower_bound<float>(sp.lifespan_multiplier, sp.lifespan_multiplier, "float",
                               ui.le_lifespan_multiplier, 0, "0");
    engine.reinit_organisms();
}

void MainWindow::le_look_range_slot() {
    le_slot_lower_bound<int>(sp.look_range, sp.look_range, "int",
                             ui.le_look_range, 1, "1");
}

void MainWindow::le_auto_food_drop_rate_slot() {
    le_slot_lower_bound<int>(sp.auto_produce_n_food, sp.auto_produce_n_food, "int",
                             ui.le_auto_produce_n_food, 0, "0");
}

void MainWindow::le_auto_produce_food_every_n_tick_slot() {
    le_slot_lower_bound<int>(sp.auto_produce_food_every_n_ticks, sp.auto_produce_food_every_n_ticks, "int",
                             ui.le_auto_produce_food_every_n_tick, 0, "0");
}

void MainWindow::le_extra_reproduction_cost_slot() {
    le_slot_no_bound<float>(sp.extra_reproduction_cost, sp.extra_reproduction_cost, "float", ui.le_extra_reproduction_cost);
}

void MainWindow::le_global_anatomy_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(sp.global_anatomy_mutation_rate, sp.global_anatomy_mutation_rate, "float",
                                     ui.le_global_anatomy_mutation_rate, 0, "0", 1, "1");
}

void MainWindow::le_global_brain_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(sp.global_brain_mutation_rate, sp.global_brain_mutation_rate, "float",
                                     ui.le_global_brain_mutation_rate, 0, "0", 1, "1");
}

void MainWindow::le_add_cell_slot() {
    le_slot_lower_bound<int>(sp.add_cell, sp.add_cell, "int",
                             ui.le_add, 0, "0");
}

void MainWindow::le_change_cell_slot() {
    le_slot_lower_bound<int>(sp.change_cell, sp.change_cell, "int",
                             ui.le_change, 0, "0");
}

void MainWindow::le_remove_cell_slot() {
    le_slot_lower_bound<int>(sp.remove_cell, sp.remove_cell, "int",
                             ui.le_remove, 0, "0");
}

void MainWindow::le_min_reproducing_distance_slot() {
    le_slot_lower_upper_bound<int>(sp.min_reproducing_distance, sp.min_reproducing_distance, "int",
                                   ui.le_min_reproduction_distance, 0, "0",
                                   sp.max_reproducing_distance, "max reproducing distance");
}

void MainWindow::le_max_reproducing_distance_slot() {
    le_slot_lower_lower_bound<int>(sp.max_reproducing_distance, sp.max_reproducing_distance, "int",
                                   ui.le_max_reproduction_distance, 1, "1",
                                   sp.min_reproducing_distance, "min reproducing distance");
}

void MainWindow::le_max_organisms_slot() {
    le_slot_no_bound<int>(edc.max_organisms, edc.max_organisms, "int", ui.le_max_organisms);
}

void MainWindow::le_float_number_precision_slot() {
    le_slot_lower_bound<int>(float_precision, float_precision, "int",
                             ui.le_float_number_precision, 0, "0");
}

void MainWindow::le_killer_damage_amount_slot() {
    le_slot_lower_bound<float>(sp.killer_damage_amount, sp.killer_damage_amount, "float",
                               ui.le_killer_damage_amount, 0, "0");
}
void MainWindow::le_produce_food_every_n_slot() {
    le_slot_lower_bound<int>(sp.produce_food_every_n_life_ticks, sp.produce_food_every_n_life_ticks, "int",
                             ui.le_produce_food_every_n_tick, 1, "1");
}

void MainWindow::le_anatomy_mutation_rate_delimiter_slot() {
    le_slot_lower_upper_bound<float>(sp.anatomy_mutation_rate_delimiter, sp.anatomy_mutation_rate_delimiter, "float",
                                     ui.le_anatomy_mutation_rate_delimiter, 0, "0",
                                     1, "1");
}
void MainWindow::le_brain_mutation_rate_delimiter_slot() {
    le_slot_lower_upper_bound<float>(sp.brain_mutation_rate_delimiter, sp.brain_mutation_rate_delimiter, "float",
                                     ui.le_brain_mutation_rate_delimiter, 0, "0",
                                     1, "1");
}

void MainWindow::le_font_size_slot() {
    auto _font = font();

    //font size could be set either by pixel_size or point_size. If it is set by one, the other will give -1.
    //so the program needs to understand which mode it is
    if (font().pixelSize() < 0) {
        font_size = font().pointSize();
        uses_point_size = true;
    } else {
        font_size = font().pixelSize();
        uses_point_size = false;
    }

    int fallback = font_size;
    auto result = try_convert_message_box_template<int>("Inputted text is not int", ui.le_font_size, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}

    font_size = result.result;

    if (uses_point_size) {
        _font.setPointSize(result.result);
    } else {
        _font.setPixelSize(result.result);
    }

    apply_font_to_windows(_font);
}

void MainWindow::le_max_move_range_slot() {
    le_slot_lower_lower_bound<int>(sp.max_move_range, sp.max_move_range, "int",
                                   ui.le_max_move_range, 1, "1",
                                   sp.min_move_range, "min move distance");
}

void MainWindow::le_min_move_range_slot() {
    le_slot_lower_upper_bound<int>(sp.min_move_range, sp.min_move_range, "int",
                                   ui.le_min_move_range, 1, "1",
                                   sp.max_move_range, "max move distance");
}

void MainWindow::le_move_range_delimiter_slot() {
    le_slot_lower_upper_bound<float>(sp.move_range_delimiter, sp.move_range_delimiter, "float",
                                     ui.le_move_range_delimiter, 0, "0",
                                     1, "1");
}

void MainWindow::le_brush_size_slot() {
    le_slot_lower_bound<int>(brush_size, brush_size, "int",
                             ui.le_brush_size, 1, "1");
}

void MainWindow::le_update_info_every_n_milliseconds_slot() {
    le_slot_lower_bound<int>(update_info_every_n_milliseconds, update_info_every_n_milliseconds, "int",
                             ui.le_update_info_every_n_milliseconds, 1, "1");
}

void MainWindow::le_menu_height_slot() {
    int fallback = ui.menu_frame->frameSize().height();
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", ui.le_menu_height, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 200) {display_message("Input cannot be less than 200."); return;}
    ui.menu_frame->setFixedHeight(result.result);}

void MainWindow::le_perlin_octaves_slot() {
    le_slot_lower_bound<int>(sp.perlin_octaves, sp.perlin_octaves, "int",
                             ui.le_perlin_octaves, 1, "1");
}

void MainWindow::le_perlin_persistence_slot() {
    le_slot_lower_upper_bound<float>(sp.perlin_persistence, sp.perlin_persistence, "float",
                                     ui.le_perlin_persistence, 0, "0",
                                     1, "1");
}

void MainWindow::le_perlin_upper_bound_slot() {
    le_slot_lower_upper_bound<float>(sp.perlin_upper_bound, sp.perlin_upper_bound, "float",
                                     ui.le_perlin_upper_bound, sp.perlin_lower_bound, "lower bound",
                                     1, "1");
}

void MainWindow::le_perlin_lower_bound_slot() {
    le_slot_lower_upper_bound<float>(sp.perlin_lower_bound, sp.perlin_lower_bound, "float",
                                     ui.le_perlin_lower_bound, 0, "0",
                                     sp.perlin_upper_bound, "upper bound");
}

void MainWindow::le_perlin_x_modifier_slot() {
    le_slot_lower_bound<float>(sp.perlin_x_modifier, sp.perlin_x_modifier, "float",
                               ui.le_perlin_x_modifier, 0, "0");
}

void MainWindow::le_perlin_y_modifier_slot() {
    le_slot_lower_bound<float>(sp.perlin_y_modifier, sp.perlin_y_modifier, "float",
                               ui.le_perlin_y_modifier, 0, "0");
}

void MainWindow::le_extra_mover_reproduction_cost_slot() {
    le_slot_no_bound<float>(sp.extra_mover_reproductive_cost, sp.extra_mover_reproductive_cost, "float", ui.le_extra_mover_reproduction_cost);
}

void MainWindow::le_anatomy_min_possible_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(sp.anatomy_min_possible_mutation_rate, sp.anatomy_min_possible_mutation_rate,
                                     "float", ui.le_anatomy_min_possible_mutation_rate, 0,
                                     "0", 1, "1");
}

void MainWindow::le_brain_min_possible_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(sp.brain_min_possible_mutation_rate, sp.brain_min_possible_mutation_rate, "float",
                                     ui.le_brain_min_possible_mutation_rate, 0, "0",
                                     1, "1");
}

void MainWindow::le_anatomy_mutation_rate_step_slot() {
    le_slot_lower_upper_bound<float>(sp.anatomy_mutations_rate_mutation_step, sp.anatomy_mutations_rate_mutation_step, "float",
                                     ui.le_anatomy_mutation_rate_step, 0, "0",
                                     1, "1");
}

void MainWindow::le_brain_mutation_rate_step_slot() {
    le_slot_lower_upper_bound<float>(sp.brain_mutation_rate_mutation_step, sp.brain_mutation_rate_mutation_step, "float",
                                     ui.le_brain_mutation_rate_step, 0, "0",
                                     1, "1");
}

void MainWindow::le_keyboard_movement_amount_slot() {
    le_slot_lower_bound<float>(keyboard_movement_amount, keyboard_movement_amount, "float",
                               ui.le_keyboard_movement_amount, 0, "0");
}

void MainWindow::le_scaling_coefficient_slot() {
    le_slot_lower_bound<float>(scaling_coefficient, scaling_coefficient, "float",
                               ui.le_scaling_coefficient, 1, "1");
}

void MainWindow::le_memory_allocation_strategy_modifier_slot() {
    le_slot_lower_bound<float>(edc.stc.memory_allocation_strategy_modifier, edc.stc.memory_allocation_strategy_modifier, "float",
                               ui.le_memory_allocation_strategy_modifier, 1, "1");
}

void MainWindow::le_random_seed_slot() {
    uint64_t temp = 0;
    le_slot_lower_bound<uint64_t>(temp, temp, "uint64_t", ui.le_random_seed, 0, "0");
    if (!temp) { return;}

    engine.set_seed(temp);
}

void MainWindow::le_set_ups_slot() {
    le_slot_lower_bound(max_ups, max_ups, "int", ui.le_set_ups, 10, "10");
    set_image_creator_interval(1./image_creation_interval+1);
}

//==================== Radio button ====================

void MainWindow::rb_food_slot() {
    set_cursor_mode(CursorMode::ModifyFood);
    ee.ui.rb_null_button->setChecked(true);
}

void MainWindow::rb_wall_slot() {
    set_cursor_mode(CursorMode::ModifyWall);
    ee.ui.rb_null_button->setChecked(true);
}

void MainWindow::rb_kill_slot() {
    set_cursor_mode(CursorMode::KillOrganism);
    ee.ui.rb_null_button->setChecked(true);
}

void MainWindow::rb_single_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Single_Threaded);
}

void MainWindow::rb_multi_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Multi_Threaded);
}

void MainWindow::rb_partial_multi_thread_slot() {
    set_simulation_mode(SimulationModes::CPU_Partial_Multi_threaded);
}

void MainWindow::rb_cuda_slot() {
    set_simulation_mode(SimulationModes::GPU_CUDA_mode);
}

//==================== Check buttons ====================

void MainWindow::cb_use_evolved_anatomy_mutation_rate_slot(bool state) {
    sp.use_anatomy_evolved_mutation_rate = state;
    ui.le_global_anatomy_mutation_rate->setDisabled(state);
}

void MainWindow::cb_use_evolved_brain_mutation_rate_slot(bool state) {
    sp.use_brain_evolved_mutation_rate = state;
    ui.le_global_brain_mutation_rate->setDisabled(state);
}

void MainWindow::cb_fill_window_slot(bool state) {
    fill_window = state;
    if (!state) {
        le_simulation_width_slot();
        le_simulation_height_slot();
    }
}

void MainWindow::cb_use_nvidia_for_image_generation_slot(bool state) {
    if (!state) {
        use_cuda = false;
#ifdef __CUDA_USED__
        cuda_creator.free();
        ee.cuda_image_creator.free();
#endif
        return;}

    if (!cuda_is_available_var) {
        ui.cb_use_nvidia_for_image_generation->setChecked(false);
        use_cuda = false;
        if (!disable_warnings) {
            display_message("Warning, CUDA is not available on this device.");
        }
        return;
    }
    use_cuda = true;
#ifdef __CUDA_USED__
    cuda_creator.copy_textures(textures);
    ee.cuda_image_creator.copy_textures(textures);
#endif
}

void MainWindow::cb_show_extended_statistics_slot(bool state) {
    if (state) {
        st.ui.lb_child_organisms->show();
        st.ui.lb_child_organisms_capacity->show();
        st.ui.lb_child_organisms_in_use->show();
        st.ui.lb_dead_organisms->show();
        st.ui.lb_organisms_capacity->show();
        st.ui.lb_total_organisms->show();
        st.ui.lb_last_alive_position->show();
        st.ui.lb_dead_inside->show();
        st.ui.lb_dead_outside->show();
        st.ui.lb_zoom->show();
        st.ui.lb_viewpoint_x->show();
        st.ui.lb_viewpoint_y->show();
    } else {
        st.ui.lb_child_organisms->hide();
        st.ui.lb_child_organisms_capacity->hide();
        st.ui.lb_child_organisms_in_use->hide();
        st.ui.lb_dead_organisms->hide();
        st.ui.lb_organisms_capacity->hide();
        st.ui.lb_total_organisms->hide();
        st.ui.lb_last_alive_position->hide();
        st.ui.lb_dead_inside->hide();
        st.ui.lb_dead_outside->hide();
        st.ui.lb_zoom->hide();
        st.ui.lb_viewpoint_x->hide();
        st.ui.lb_viewpoint_y->hide();
    }
}

void MainWindow::cb_statistics_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = st.isHidden();

    st.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    //Why sleep? For some reason changes are not applied instantly and main process needs to wait for some time.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        st.show();
    }
}

void MainWindow::cb_editor_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = ee.isHidden();

    ee.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        ee.show();
    }
}

void MainWindow::cb_info_window_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = iw.isHidden();

    iw.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        iw.show();
    }
}

void MainWindow::cb_recorder_window_always_on_top_slot(bool state) {
#ifndef __NO_RECORDER__
    if (is_fullscreen) {return;}
    auto hidden = rc.isHidden();

    rc.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        rc.show();
    }
#endif
}

void MainWindow::cb_world_events_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = we.isHidden();

    we.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        we.show();
    }
}

void MainWindow::cb_benchmarks_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = bs.isHidden();

    bs.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        bs.show();
    }
}

void MainWindow::cb_occp_always_on_top_slot(bool state) {
    if (is_fullscreen) {return;}
    auto hidden = occpw.isHidden();

    occpw.setWindowFlag(Qt::WindowStaysOnTopHint, state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hidden) {
        occpw.show();
    }
}

void MainWindow::cb_really_stop_render_slot(bool state) {
    really_stop_render = state;
    if (!state && pause_grid_parsing) {
        parse_full_simulation_grid(true);
    }
}

void MainWindow::cb_use_occ_slot(bool state) {
    if (state == sp.use_occ) { return;}
    if (!display_dialog_message("Changing this parameter will reset world. Continue?", disable_warnings)) {
        ui.cb_use_organism_construction_code->setChecked(sp.use_occ);
        return;
    }

    ee.occ_mode(state);

    if (state) {
        st.ui.lb_avg_occ_len_2->show();
        st.ui.lb_avg_occ_len_3->show();
        st.ui.lb_avg_occ_length_4->show();
        st.ui.lb_total_occ_len_2->show();
        st.ui.lb_total_occ_length_3->show();
        st.ui.lb_total_occ_length_4->show();
    } else {
        st.ui.lb_avg_occ_len_2->hide();
        st.ui.lb_avg_occ_len_3->hide();
        st.ui.lb_avg_occ_length_4->hide();
        st.ui.lb_total_occ_len_2->hide();
        st.ui.lb_total_occ_length_3->hide();
        st.ui.lb_total_occ_length_4->hide();
    }

    engine.pause();
    sp.use_occ = state;
    engine.reset_world();
    engine.unpause();
}

void MainWindow::cb_recenter_to_imaginary_slot(bool state) {
    if (state == sp.recenter_to_imaginary_pos) { return;}

    engine.pause();

    sp.recenter_to_imaginary_pos = state;
    engine.reinit_organisms();

    engine.unpause();
}

void MainWindow::cb_use_weighted_brain_slot(bool state) {
    sp.use_weighted_brain = state;
    engine.reinit_organisms();
    ee.update_brain_edit_visibility(state);
}

void MainWindow::cb_reproduction_rotation_enabled_slot   (bool state) { sp.reproduction_rotation_enabled = state;}

void MainWindow::cb_on_touch_kill_slot                   (bool state) { sp.on_touch_kill = state;}

void MainWindow::cb_movers_can_produce_food_slot         (bool state) { sp.movers_can_produce_food = state;}

void MainWindow::cb_food_blocks_reproduction_slot        (bool state) { sp.food_blocks_reproduction = state;}

void MainWindow::cb_reset_on_total_extinction_slot       (bool state) { sp.reset_on_total_extinction = state;}

void MainWindow::cb_pause_on_total_extinction_slot       (bool state) { sp.pause_on_total_extinction = state;}

void MainWindow::cb_clear_walls_on_reset_slot            (bool state) { sp.clear_walls_on_reset = state;}

void MainWindow::cb_generate_random_walls_on_reset_slot  (bool state) { sp.generate_random_walls_on_reset = state;}

void MainWindow::cb_runtime_rotation_enabled_slot        (bool state) { sp.runtime_rotation_enabled = state;}

void MainWindow::cb_fix_reproduction_distance_slot       (bool state) { sp.reproduction_distance_fixed = state;}

void MainWindow::cb_disable_warnings_slot                (bool state) { disable_warnings = state;}

void MainWindow::cb_set_fixed_move_range_slot            (bool state) { sp.set_fixed_move_range = state;}

void MainWindow::cb_self_organism_blocks_block_sight_slot(bool state){ sp.organism_self_blocks_block_sight = state;}

void MainWindow::cb_failed_reproduction_eats_food_slot   (bool state) { sp.failed_reproduction_eats_food = state;}

void MainWindow::cb_rotate_every_move_tick_slot          (bool state) { sp.rotate_every_move_tick = state;}

void MainWindow::cb_multiply_food_production_prob_slot   (bool state) { sp.multiply_food_production_prob = state; engine.reinit_organisms();}

void MainWindow::cb_simplified_food_production_slot      (bool state) { sp.simplified_food_production = state;}

void MainWindow::cb_stop_when_one_food_generated         (bool state) { sp.stop_when_one_food_generated = state;}

void MainWindow::cb_synchronise_info_with_window_slot    (bool state) { synchronise_info_update_with_window_update = state;}

void MainWindow::cb_eat_then_produce_slot                (bool state) { sp.eat_then_produce = state;}

void MainWindow::cb_food_blocks_movement_slot            (bool state) { sp.food_blocks_movement = state;}

void MainWindow::cb_use_new_child_pos_calculator_slot    (bool state) { sp.use_new_child_pos_calculator = state;}

void MainWindow::cb_check_if_path_is_clear_slot          (bool state) { sp.check_if_path_is_clear = state;}

void MainWindow::cb_reset_with_editor_organism_slot      (bool state) { ecp.reset_with_editor_organism = state;}

void MainWindow::cb_no_random_decisions_slot             (bool state) { sp.no_random_decisions = state;}

void MainWindow::cb_do_not_mutate_brain_of_plants_slot   (bool state) { sp.do_not_mutate_brains_of_plants = state;}

void MainWindow::cb_load_evolution_controls_from_state_slot(bool state) { save_simulation_settings = state;}

//==================== Table ====================

void MainWindow::table_cell_changed_slot(int row, int col) {
    auto item = ui.table_organism_block_parameters->item(row, col);
    float result;
    bool set_result = false;
    if (boost::conversion::try_lexical_convert<float>(item->text().toStdString(), result)) {
        set_result = true;
    } else {
        if (!disable_warnings) {display_message("Value should be float.");}
    }
    BParameters * type;
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
        case ParametersNames::LifetimeWeight:   value = &type->lifetime_weight;    break;
        case ParametersNames::ChanceWeight:     value = &type->chance_weight;      break;
    }

    if(set_result) {*value = result; engine.reinit_organisms(); return;}

    ui.table_organism_block_parameters->item(row, col)->setText(QString::fromStdString(to_str(*value)));
    ui.table_organism_block_parameters->update();

    engine.reinit_organisms();
}