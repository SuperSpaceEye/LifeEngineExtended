// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.03.2022.
//

#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent):
        QWidget(parent){
    ui.setupUi(this);

    ui.simulation_graphicsView->show();
    //If not false, the graphics view allows scrolling of an image after window resizing and only this helps.
    //Disabling graphics view doesn't change anything anyway.
    ui.simulation_graphicsView->setEnabled(false);

    //https://stackoverflow.com/questions/32714105/mousemoveevent-is-not-called
    QCoreApplication::instance()->installEventFilter(this);

    edc.simulation_width = 200;
    edc.simulation_height = 200;

    edc.st_grid.resize(edc.simulation_width, edc.simulation_height);
    edc.simple_state_grid.resize(edc.simulation_width * edc.simulation_height, BaseGridBlock{});

    update_simulation_size_label();

    engine.make_walls();

    #ifndef __NO_RECORDER__
    rc.set_engine(&engine);
    #endif

    //In mingw compiler std::random_device is deterministic?
    //https://stackoverflow.com/questions/18880654/why-do-i-get-the-same-sequence-for-every-run-with-stdrandom-device-with-mingw

    cc = ColorContainer{};
    sp = SimulationParameters{};
//    sp.growth_of_organisms = false;

    cuda_is_available_var = cuda_is_available();

#ifndef __NO_RECORDER__
    rc.init_gui();
#endif

    auto anatomy = Anatomy();

    auto brain = Brain(BrainTypes::RandomActions);

    auto occ = OrganismConstructionCode();
    occ.set_code(std::vector<OCCInstruction>{OCCInstruction::SetBlockMouth,
                                             OCCInstruction::ShiftUpLeft, OCCInstruction::SetBlockProducer,
                                             OCCInstruction::ShiftDownRight, OCCInstruction::SetBlockProducer});

    if (sp.use_occ) {
        anatomy = Anatomy(occ.compile_code(edc.stc.occl, sp.growth_of_organisms));
    } else {
        anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
        anatomy.set_block(BlockTypes::ProducerBlock, Rotation::UP, -1, -1);
        anatomy.set_block(BlockTypes::ProducerBlock, Rotation::UP, 1, 1);
    }

    ee.occ_mode(sp.use_occ);

    edc.base_organism = Organism(edc.simulation_width / 2, edc.simulation_height / 2,
                                 Rotation::UP, anatomy, brain, occ, &sp, &bp, &occp,
                                 &edc.stc.occl, 1, 0.05, 0.1, false);
    edc.chosen_organism = Organism(edc.simulation_width / 2, edc.simulation_height / 2,
                                   Rotation::UP, Anatomy(anatomy), Brain(), OrganismConstructionCode(occ), &sp, &bp,
                                   &occp, &edc.stc.occl, 1, 0.05, 0.1, false);

    edc.base_organism.last_decision_observation = DecisionObservation{};
    edc.chosen_organism.last_decision_observation = DecisionObservation{};

    auto * organism = OrganismsController::get_new_main_organism(edc);
    auto array_place = organism->vector_index;
    organism->copy_organism(edc.base_organism);
    organism->vector_index = array_place;

    SimulationEngineSingleThread::place_organism(edc, *organism, sp);

    resize_image(image_width, image_height);
    reset_scale_view();

    //Will execute on first QT show event
    QTimer::singleShot(0, [&]{
        engine_thread = std::thread{&SimulationEngine::threaded_mainloop, std::ref(engine)};
        engine_thread.detach();

        create_image_creation_thread();

        fps_timer = clock_now();

        scene.addItem(&pixmap_item);
        ui.simulation_graphicsView->setScene(&scene);

        timer->start();

        reset_scale_view();
        get_current_font_size();
        initialize_gui();
        #if defined(__WIN32)
        ShowWindow(GetConsoleWindow(), SW_HIDE);
        #endif

        load_state();

        le_update_info_every_n_milliseconds_slot();
    });

    timer = new QTimer(parent);
    //Will execute as fast as possible
    connect(timer, &QTimer::timeout, [&]{mainloop_tick();});

    set_image_creator_interval(60);
    set_simulation_interval(60);

    cb_show_extended_statistics_slot(false);

    auto executable_path = QCoreApplication::applicationDirPath().toStdString();
    if (!std::filesystem::exists(executable_path + "/temp")) {
        std::filesystem::create_directory(executable_path + "/temp");
    }

    if (!std::filesystem::exists(executable_path + "/videos")) {
        std::filesystem::create_directory(executable_path + "/videos");
    }

    if (!std::filesystem::exists(executable_path + "/textures")) {
        std::filesystem::create_directory(executable_path + "/textures");
    }

    load_textures_from_disk();

    //TODO
    ui.tb_open_benchmarks->setEnabled(false);
    ui.tb_open_benchmarks->hide();
    ui.cb_benchmarks_always_on_top->hide();
    //biome buttons
    ui.pushButton->hide();
    ui.checkBox->hide();
}

void MainWindow::mainloop_tick() {
    ui_tick();

    auto info_update = std::chrono::duration_cast<std::chrono::microseconds>(clock_now() - fps_timer).count();

    if (synchronise_info_update_with_window_update || info_update >= update_info_every_n_milliseconds*1000) {
        uint32_t simulation_frames = edc.engine_ticks_between_updates;
        edc.engine_ticks_between_updates = 0;

        if (info_update == 0) {info_update = 1;}

        auto scale = (info_update/1000000.);

        last_fps = last_fps * 0.5 + (image_frames / scale) * 0.5;
        last_sps = last_sps * 0.5 + (simulation_frames / scale) * 0.5;
        last_ups = last_ups * 0.5 + (window_frames / scale) * 0.5;

        image_frames = 0;

        update_fps_labels(last_fps, last_sps, last_ups);
        window_frames = 0;
        fps_timer = clock_now();

        engine.update_info();
        auto info = engine.get_info();

        update_statistics_info(info);
    }
}

void MainWindow::ui_tick() {
    if (!ecp.engine_working && !engine_error) {
        display_message("Simulation Engine has stopped unexpectedly.");
        engine_error = true;
    }

    if (ecp.update_editor_organism) { ee.load_chosen_organism(); ecp.update_editor_organism = false;}

    if (resize_simulation_grid_flag) { resize_simulation_grid(); resize_simulation_grid_flag=false;}
    if (ecp.tb_paused) {ui.tb_pause->setChecked(true);}

    if (left_mouse_button_pressed  && change_main_simulation_grid) {change_main_grid_left_click();}
    if (right_mouse_button_pressed && change_main_simulation_grid) {change_main_grid_right_click();}

    if (left_mouse_button_pressed  && change_editing_grid) {change_editing_grid_left_click();}
    if (right_mouse_button_pressed && change_editing_grid) {change_editing_grid_right_click();}

    if (ecp.execute_world_events && (!ecp.tb_paused && !ecp.pause_world_events) || ecp.update_world_events_ui_once) {update_world_event_values_ui(); ecp.update_world_events_ui_once = false;}

    if (update_textures) {load_textures_from_disk(); update_textures = false;}

    bs.update_();

#ifndef __NO_RECORDER__
    rc.update_label();
#endif

    window_frames++;

    if (pause_grid_parsing && really_stop_render) { return;}

    if (do_not_parse_image_data_ct) { return;}
    do_not_parse_image_data_mt.store(true);
    pixmap_item.setPixmap(QPixmap::fromImage(QImage(image_vectors[ready_buffer].data(), image_width, image_height, QImage::Format_RGB32).copy()));
    have_read_buffer = true;
    do_not_parse_image_data_mt.store(false);
}

void MainWindow::update_fps_labels(int fps, int tps, int ups) {
    ui.lb_fps->setText(QString::fromStdString("fps: " + std::to_string(fps)));
    ui.lb_sps->setText(QString::fromStdString("tps: " + std::to_string(tps)));
    ui.lb_ups->setText(QString::fromStdString("ups: " + std::to_string(ups)));
}

void MainWindow::resize_image(int image_width, int image_height) {
    if (image_vectors[0].size() == 4 * image_width * image_height) {return;}
    image_vectors[0].resize(4 * image_width * image_height, 255);
    image_vectors[1].resize(4 * image_width * image_height, 255);
}

void MainWindow::move_center(int delta_x, int delta_y) {
    if (change_main_simulation_grid) {
        center_x -= delta_x * scaling_zoom;
        center_y -= delta_y * scaling_zoom;
    } else {
        ee.move_center(delta_x, delta_y);
    }
}

void MainWindow::reset_scale_view() {
    float exp;
    center_x = (float)edc.simulation_width / 2;
    center_y = (float)edc.simulation_height / 2;
    // finds exponent needed to scale the image

    //if simulation dimensions are equal, then exponent depends on window size
    if (edc.simulation_width == edc.simulation_height) {
        if (ui.simulation_graphicsView->viewport()->height() < ui.simulation_graphicsView->viewport()->width()) {
            exp = log((float) edc.simulation_height / (float) ui.simulation_graphicsView->viewport()->height()) /
                  log(scaling_coefficient);
        } else {
            exp = log((float) edc.simulation_width / (float) ui.simulation_graphicsView->viewport()->width()) /
                  log(scaling_coefficient);
        }
    // if not equal, then to capture full view you need to scale by largest dimension
    } else {
        if (edc.simulation_width > edc.simulation_height) {
            exp = log((float) edc.simulation_width / (float) ui.simulation_graphicsView->viewport()->width()) /
                  log(scaling_coefficient);
        } else {
            exp = log((float) edc.simulation_height / (float) ui.simulation_graphicsView->viewport()->height()) /
                  log(scaling_coefficient);
        }
    }
    scaling_zoom = pow(scaling_coefficient, exp);
}


void MainWindow::create_image() {
    while (do_not_parse_image_data_mt) {}
    do_not_parse_image_data_ct.store(true);

    std::vector<int> lin_width;
    std::vector<int> lin_height;
    std::vector<int> truncated_lin_width;
    std::vector<int> truncated_lin_height;

    pre_parse_simulation_grid_stage(lin_width, lin_height, truncated_lin_width, truncated_lin_height);

    parse_simulation_grid_stage(truncated_lin_width, truncated_lin_height);

    int new_buffer = !bool(ready_buffer);

#ifdef __CUDA_USED__
    void * cuda_creator_ptr = &cuda_creator;
#else
    void * cuda_creator_ptr = nullptr;
#endif

    ImageCreation::create_image(lin_width, lin_height, edc.simulation_width, edc.simulation_height, cc, textures,
                                image_width, image_height, image_vectors[new_buffer], edc.simple_state_grid,
                                use_cuda, cuda_is_available_var, cuda_creator_ptr, truncated_lin_width,
                                truncated_lin_height, false, 1);

    ready_buffer = new_buffer;
    do_not_parse_image_data_ct.store(false);
}

void MainWindow::parse_simulation_grid_stage(const std::vector<int> &truncated_lin_width,
                                             const std::vector<int> &truncated_lin_height) {
    if (!pause_grid_parsing && !ecp.engine_global_pause) {
        parse_simulation_grid(truncated_lin_width, truncated_lin_height);
    }
}

void MainWindow::pre_parse_simulation_grid_stage(std::vector<int> &lin_width, std::vector<int> &lin_height,
                                                 std::vector<int> &truncated_lin_width,
                                                 std::vector<int> &truncated_lin_height) {
    image_width  = ui.simulation_graphicsView->viewport()->width();
    image_height = ui.simulation_graphicsView->viewport()->height();
    resize_image(image_width, image_height);
    int scaled_width  = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and y coordinates on simulation grid
    auto start_x = int(center_x - (scaled_width / 2));
    auto end_x   = int(center_x + (scaled_width / 2));

    auto start_y = int(center_y - (scaled_height / 2));
    auto end_y   = int(center_y + (scaled_height / 2));
    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    truncated_lin_width .reserve(std::abs(lin_width [lin_width.size() -1])+1);
    truncated_lin_height.reserve(std::abs(lin_height[lin_height.size()-1])+1);

    ImageCreation::calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);
}

void MainWindow::set_image_creator_interval(int max_window_fps) {
    if (max_window_fps <= 0) {
        image_creation_interval = 0.;
        timer->setInterval(1000./max_ups);
        return;
    }
    image_creation_interval = 1. / max_window_fps;
    if (max_window_fps > max_ups) {
        timer->setInterval(1000./max_window_fps);
    } else {
        timer->setInterval(1000./max_ups);
    }
}

void MainWindow::set_simulation_interval(int max_simulation_fps) {
    if (max_simulation_fps <= 0) {
        edc.simulation_interval = 0.;
        edc.unlimited_simulation_fps = true;
        return;
    }
    edc.simulation_interval = 1. / max_simulation_fps;
    edc.unlimited_simulation_fps = false;
}


void MainWindow::parse_simulation_grid(const std::vector<int> &lin_width, const std::vector<int> &lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= edc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= edc.simulation_height) { continue; }
            auto type = edc.st_grid.get_type(x, y);
            auto & simple_block = edc.simple_state_grid[x + y * edc.simulation_width];
            simple_block.type = type;
            simple_block.rotation = edc.st_grid.get_rotation(x, y);

            if (type == BlockTypes::EmptyBlock && edc.st_grid.get_food_num(x, y) >= sp.food_threshold) {
                simple_block.type = BlockTypes::FoodBlock;}
        }
    }
}

void MainWindow::parse_full_simulation_grid(bool parse) {
    if (!parse) {return;}
    engine.pause();
    engine.parse_full_simulation_grid();
    engine.unpause();
}

void MainWindow::set_simulation_num_threads(uint8_t num_threads) {
    engine.pause();

    ecp.num_threads = num_threads;
    if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        ecp.build_threads = true;
    }

    engine.unpause();
}

void MainWindow::set_cursor_mode(CursorMode mode) {
    cursor_mode = mode;
}

void MainWindow::set_simulation_mode(SimulationModes mode) {
    ecp.change_to_mode = mode;
    ecp.change_simulation_mode = true;
}

void MainWindow::calculate_new_simulation_size() {
    auto window_size = ui.simulation_graphicsView->viewport()->size();

    new_simulation_width  = window_size.width() / starting_cell_size_on_resize;
    new_simulation_height = window_size.height() / starting_cell_size_on_resize;
}

void MainWindow::just_resize_simulation_grid() {
    edc.simulation_width = new_simulation_width;
    edc.simulation_height = new_simulation_height;

    edc.st_grid.clear();
    edc.simple_state_grid.clear();

#ifdef __CUDA_USED__
    if (cuda_is_available_var && use_cuda) {
        cuda_creator.device_state_grid.clear();
        cuda_creator.device_state_grid.resize(edc.simulation_width*edc.simulation_height);
    }
#endif

    edc.st_grid.resize(edc.simulation_width, edc.simulation_height);
    edc.simple_state_grid.resize(edc.simulation_width * edc.simulation_height, BaseGridBlock{});

    engine.init_auto_food_drop(edc.simulation_width, edc.simulation_height);
}

void MainWindow::resize_simulation_grid() {
    if (ecp.lock_resizing) {
        display_message("Grid cannot be resized until recording is stopped.");
        return;
    }
    if (fill_window) {calculate_new_simulation_size();}

    if (!disable_warnings) {
        if (!use_cuda) {
            auto msg = DescisionMessageBox("Warning",
                                                   QString::fromStdString("Simulation space will be rebuilt and all organisms cleared.\n"
                                       "New grid will need " + convert_num_bytes((sizeof(BaseGridBlock) * 2 + sizeof(int32_t) * 2) * new_simulation_height * new_simulation_width)),
                                                   "OK", "Cancel", this);
            auto result = msg.exec();
            if (!result) {
                return;
            }
        } else {
            auto msg = DescisionMessageBox("Warning",
                                           QString::fromStdString("Simulation space will be rebuilt and all organisms cleared.\n"
                                                                  "New grid will need " + convert_num_bytes((sizeof(BaseGridBlock) * 2 + sizeof(int32_t) * 2) * new_simulation_height * new_simulation_width)
                                                                  + " of RAM and " + convert_num_bytes(sizeof(BaseGridBlock) * new_simulation_height * new_simulation_width))
                                                                  + " GPU's VRAM",
                                                   "OK", "Cancel", this);
            auto result = msg.exec();
            if (!result) {
                return;
            }
        }
    }

    engine.pause();

    just_resize_simulation_grid();

    if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        ecp.build_threads = true;
    }

    engine.reset_world();
    engine.unpause();

    update_simulation_size_label();

    reset_scale_view();
}

void MainWindow::clear_world() {
    engine.partial_clear_world();
    engine.make_walls();
    engine.unpause();
}

void MainWindow::update_statistics_info(const OrganismInfoContainer &info) {
    st.update_statistics(info, edc, float_precision, scaling_zoom, center_x, center_y);
}

// So that changes in code values would be set by default in gui.
void MainWindow::initialize_gui() {
    auto temp = disable_warnings;
    disable_warnings = true;

    //World settings
    ui.le_cell_size         ->setText(QString::fromStdString(std::to_string(starting_cell_size_on_resize)));
    ui.le_simulation_width  ->setText(QString::fromStdString(std::to_string(edc.simulation_width)));
    ui.le_simulation_height ->setText(QString::fromStdString(std::to_string(edc.simulation_height)));
    ui.le_max_organisms     ->setText(QString::fromStdString(std::to_string(edc.max_organisms)));
    ui.le_brush_size        ->setText(QString::fromStdString(std::to_string(brush_size)));
    ui.cb_reset_on_total_extinction ->setChecked(sp.reset_on_total_extinction);
    ui.cb_pause_on_total_extinction ->setChecked(sp.pause_on_total_extinction);
    ui.cb_fill_window               ->setChecked(fill_window);
    //Evolution settings
    ui.le_food_production_probability       ->setText(QString::fromStdString(to_str(sp.food_production_probability, 4)));
    ui.le_global_anatomy_mutation_rate      ->setText(QString::fromStdString(to_str(sp.global_anatomy_mutation_rate, 2)));
    ui.le_global_brain_mutation_rate        ->setText(QString::fromStdString(to_str(sp.global_brain_mutation_rate, 2)));
    ui.le_anatomy_mutation_rate_delimiter   ->setText(QString::fromStdString(to_str(sp.anatomy_mutation_rate_delimiter, 2)));
    ui.le_brain_mutation_rate_delimiter     ->setText(QString::fromStdString(to_str(sp.brain_mutation_rate_delimiter, 2)));
    ui.le_move_range_delimiter              ->setText(QString::fromStdString(to_str(sp.move_range_delimiter, 2)));
    ui.le_lifespan_multiplier               ->setText(QString::fromStdString(to_str(sp.lifespan_multiplier, 3)));
    ui.le_brain_min_possible_mutation_rate  ->setText(QString::fromStdString(to_str(sp.brain_min_possible_mutation_rate, 3)));
    ui.le_anatomy_min_possible_mutation_rate->setText(QString::fromStdString(to_str(sp.anatomy_min_possible_mutation_rate, 3)));
    ui.le_extra_mover_reproduction_cost     ->setText(QString::fromStdString(to_str(sp.extra_mover_reproductive_cost, 0)));
    ui.le_extra_reproduction_cost           ->setText(QString::fromStdString(to_str(sp.extra_reproduction_cost, 0)));
    ui.le_anatomy_mutation_rate_step        ->setText(QString::fromStdString(to_str(sp.anatomy_mutations_rate_mutation_step, 2)));
    ui.le_brain_mutation_rate_step          ->setText(QString::fromStdString(to_str(sp.brain_mutation_rate_mutation_step, 2)));
    ui.le_continuous_movement_drag          ->setText(QString::fromStdString(to_str(sp.continuous_movement_drag, 2)));
    ui.le_food_threshold                    ->setText(QString::fromStdString(to_str(sp.food_threshold, 2)));
    ui.le_max_food                          ->setText(QString::fromStdString(to_str(sp.max_food)));
    ui.le_cell_growth_modifier              ->setText(QString::fromStdString(to_str(sp.cell_growth_modifier)));
    ui.le_produce_food_every_n_tick         ->setText(QString::fromStdString(std::to_string(sp.produce_food_every_n_life_ticks)));
    ui.le_look_range                        ->setText(QString::fromStdString(std::to_string(sp.look_range)));
    ui.le_auto_produce_n_food               ->setText(QString::fromStdString(std::to_string(sp.auto_produce_n_food)));
    ui.le_auto_produce_food_every_n_tick    ->setText(QString::fromStdString(std::to_string(sp.auto_produce_food_every_n_ticks)));
    ui.le_add                               ->setText(QString::fromStdString(std::to_string(sp.add_cell)));
    ui.le_change                            ->setText(QString::fromStdString(std::to_string(sp.change_cell)));
    ui.le_remove                            ->setText(QString::fromStdString(std::to_string(sp.remove_cell)));
    ui.le_min_reproduction_distance         ->setText(QString::fromStdString(std::to_string(sp.min_reproducing_distance)));
    ui.le_max_reproduction_distance         ->setText(QString::fromStdString(std::to_string(sp.max_reproducing_distance)));
    ui.le_min_move_range                    ->setText(QString::fromStdString(std::to_string(sp.min_move_range)));
    ui.le_max_move_range                    ->setText(QString::fromStdString(std::to_string(sp.max_move_range)));
    ui.le_starting_organism_size            ->setText(QString::fromStdString(std::to_string(sp.starting_organism_size)));

    ui.cb_reproducing_rotation_enabled      ->setChecked(sp.reproduction_rotation_enabled);
    ui.cb_runtime_rotation_enabled          ->setChecked(sp.runtime_rotation_enabled);
    ui.cb_on_touch_kill                     ->setChecked(sp.on_touch_kill);
    ui.cb_movers_can_produce_food           ->setChecked(sp.movers_can_produce_food);
    ui.cb_food_blocks_reproduction          ->setChecked(sp.food_blocks_reproduction);
    ui.cb_food_blocks_movement              ->setChecked(sp.food_blocks_movement);
    ui.cb_fix_reproduction_distance         ->setChecked(sp.reproduction_distance_fixed);
    ui.cb_use_evolved_brain_mutation_rate   ->setChecked(sp.use_brain_evolved_mutation_rate);
    ui.cb_use_evolved_anatomy_mutation_rate ->setChecked(sp.use_anatomy_evolved_mutation_rate);
    ui.cb_disable_warnings                  ->setChecked(disable_warnings);
    ui.cb_self_organism_blocks_block_sight  ->setChecked(sp.organism_self_blocks_block_sight);
    ui.cb_set_fixed_move_range              ->setChecked(sp.set_fixed_move_range);
    ui.cb_failed_reproduction_eats_food     ->setChecked(sp.failed_reproduction_eats_food);
    ui.cb_rotate_every_move_tick            ->setChecked(sp.rotate_every_move_tick);
    ui.cb_multiply_food_production_prob     ->setChecked(sp.multiply_food_production_prob);
    ui.cb_simplified_food_production        ->setChecked(sp.simplified_food_production);
    ui.cb_stop_when_one_food_generated      ->setChecked(sp.stop_when_one_food_generated);
    ui.cb_eat_then_produce                  ->setChecked(sp.eat_then_produce);
    ui.cb_checks_if_path_is_clear           ->setChecked(sp.check_if_path_is_clear);
    ui.cb_no_random_decisions               ->setChecked(sp.no_random_decisions);
    ui.cb_use_organism_construction_code    ->setChecked(sp.use_occ);
    ui.cb_recenter_to_imaginary             ->setChecked(sp.recenter_to_imaginary_pos);
    ui.cb_do_not_mutate_brain_of_plants     ->setChecked(sp.do_not_mutate_brains_of_plants);
    ui.cb_use_weighted_brain                ->setChecked(sp.use_weighted_brain);
    ui.cb_organisms_destroy_food            ->setChecked(sp.organisms_destroy_food);
    ui.cb_use_continuous_movement           ->setChecked(sp.use_continuous_movement);
    ui.cb_enable_organism_growth            ->setChecked(sp.growth_of_organisms);

    //Settings
    ui.le_perlin_persistence->setText(QString::fromStdString(to_str(sp.perlin_persistence, 3)));
    ui.le_perlin_upper_bound->setText(QString::fromStdString(to_str(sp.perlin_upper_bound, 3)));
    ui.le_perlin_lower_bound->setText(QString::fromStdString(to_str(sp.perlin_lower_bound, 3)));
    ui.le_perlin_x_modifier ->setText(QString::fromStdString(to_str(sp.perlin_x_modifier, 3)));
    ui.le_perlin_y_modifier ->setText(QString::fromStdString(to_str(sp.perlin_y_modifier, 3)));
    ui.le_keyboard_movement_amount->setText(QString::fromStdString(to_str(keyboard_movement_amount, 1)));
    ui.le_scaling_coefficient->setText(QString::fromStdString(to_str(scaling_coefficient, 1)));
    ui.le_memory_allocation_strategy_modifier->setText(QString::fromStdString(
            to_str(edc.stc.memory_allocation_strategy_modifier, 0)));

    ui.le_num_threads->setText(QString::fromStdString(std::to_string(ecp.num_threads)));
    ui.le_float_number_precision->setText(QString::fromStdString(std::to_string(float_precision)));
    ui.le_perlin_octaves->setText(QString::fromStdString(std::to_string(sp.perlin_octaves)));
    ui.le_font_size              ->setText(QString::fromStdString(std::to_string(font_size)));

//    ui.rb_partial_multi_thread_mode->hide();
    ui.rb_multi_thread_mode->hide();
    ui.rb_cuda_mode->hide();

    ui.table_organism_block_parameters->horizontalHeader()->setVisible(true);
    ui.table_organism_block_parameters->verticalHeader()->setVisible(true);

    ui.le_update_info_every_n_milliseconds ->setText(QString::fromStdString(std::to_string(update_info_every_n_milliseconds)));
    ui.cb_synchronise_info_with_window->setChecked(synchronise_info_update_with_window_update);
    auto state = disable_warnings;
    disable_warnings = true;
    ui.cb_use_nvidia_for_image_generation->setChecked(use_cuda);
    disable_warnings = state;
    ui.le_menu_height->setText(QString::fromStdString(std::to_string(ui.menu_frame->frameSize().height())));
    ui.cb_really_stop_render->setChecked(really_stop_render);
    ui.cb_show_extended_statistics->setChecked(false);
    ui.cb_load_evolution_controls_from_state->setChecked(save_simulation_settings);
#ifndef __CUDA_USED__
    ui.cb_use_nvidia_for_image_generation->hide();
#endif

    //So that when user clicks on rbs in organism editors, rbs in main window would be unchecked and vice versa
    ui.rb_null_button->hide();
    ee.ui.rb_null_button->hide();

    //TODO
    ui.rb_single_thread_mode->hide();
    ui.rb_partial_multi_thread_mode->hide();
    ui.le_num_threads->hide();
    ui.lb_set_num_threads->hide();
    st.ui.lb_organisms_memory_consumption->hide();

    ee.occ_mode(sp.use_occ);

    ui.le_set_ups->setText(QString::fromStdString(std::to_string(max_ups)));

    #ifdef __NO_RECORDER__
    ui.cb_recorder_window_always_on_top->hide();
    ui.tb_open_recorder_window->hide();
    #endif

    ee.update_brain_edit_visibility(sp.use_weighted_brain);

    disable_warnings = temp;
}

void MainWindow::get_current_font_size() {
    //font size could be set either by pixel_size or point_size. If it is set by one, the other will give -1
    if (font().pixelSize() < 0) {
        font_size = font().pointSize();
        uses_point_size = true;
    } else {
        font_size = font().pixelSize();
        uses_point_size = false;
    }
}

void MainWindow::update_world_event_values_ui() {
    ui.le_food_production_probability       ->setText(QString::fromStdString(
            to_str(sp.food_production_probability, 4)));
    ui.le_global_anatomy_mutation_rate      ->setText(QString::fromStdString(
            to_str(sp.global_anatomy_mutation_rate, 2)));
    ui.le_global_brain_mutation_rate        ->setText(QString::fromStdString(
            to_str(sp.global_brain_mutation_rate, 2)));
    ui.le_anatomy_mutation_rate_delimiter   ->setText(QString::fromStdString(
            to_str(sp.anatomy_mutation_rate_delimiter, 2)));
    ui.le_brain_mutation_rate_delimiter     ->setText(QString::fromStdString(
            to_str(sp.brain_mutation_rate_delimiter, 2)));
    ui.le_lifespan_multiplier               ->setText(QString::fromStdString(to_str(sp.lifespan_multiplier, 3)));
    ui.le_extra_mover_reproduction_cost     ->setText(QString::fromStdString(
            to_str(sp.extra_mover_reproductive_cost, 0)));
    ui.le_extra_reproduction_cost           ->setText(QString::fromStdString(
            to_str(sp.extra_reproduction_cost, 0)));
    ui.le_produce_food_every_n_tick         ->setText(QString::fromStdString(std::to_string(sp.produce_food_every_n_life_ticks)));
    ui.le_auto_produce_n_food               ->setText(QString::fromStdString(std::to_string(sp.auto_produce_n_food)));
    ui.le_auto_produce_food_every_n_tick    ->setText(QString::fromStdString(std::to_string(sp.auto_produce_food_every_n_ticks)));
    ui.le_add                               ->setText(QString::fromStdString(std::to_string(sp.add_cell)));
    ui.le_change                            ->setText(QString::fromStdString(std::to_string(sp.change_cell)));
    ui.le_remove                            ->setText(QString::fromStdString(std::to_string(sp.remove_cell)));

    update_table_values();
}

void MainWindow::update_simulation_size_label() {
  st.ui.lb_simulation_size->setText(QString::fromStdString("Simulation size: " + std::to_string(edc.simulation_width) + "x" + std::to_string(edc.simulation_height)));
}

Vector2<int> MainWindow::calculate_cursor_pos_on_grid(int x, int y) {
    auto c_pos = Vector2<int>{};
    c_pos.x = static_cast<int>((x - float(ui.simulation_graphicsView->viewport()->width() ) / 2) * scaling_zoom + center_x);
    c_pos.y = static_cast<int>((y - float(ui.simulation_graphicsView->viewport()->height()) / 2) * scaling_zoom + center_y);
    return c_pos;
}

void MainWindow::change_main_grid_left_click() {
    ecp.do_not_use_user_actions_ui = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    while (ecp.do_not_use_user_actions_engine) {}

    if (update_last_cursor_pos) {
        last_last_cursor_x_pos = last_mouse_x_pos;
        last_last_cursor_y_pos = last_mouse_y_pos;

        update_last_cursor_pos = false;
    }

    //cursor Vector2 on grid
    auto cpg = calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    auto cpg2 = calculate_cursor_pos_on_grid(last_last_cursor_x_pos, last_last_cursor_y_pos);

    std::vector<Vector2<int>> points;

    if (cursor_mode == CursorMode::ModifyFood || cursor_mode == CursorMode::ModifyWall || cursor_mode == CursorMode::KillOrganism) {
        points = iterate_between_two_points(cpg, cpg2);
    } else {
        points.emplace_back(cpg);
    }

    for (auto & pt: points) {
        for (int x = -brush_size / 2; x < float(brush_size) / 2; x++) {
            for (int y = -brush_size / 2; y < float(brush_size) / 2; y++) {
                switch (cursor_mode) {
                    case CursorMode::ModifyFood:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryAddFood, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::ModifyWall:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryAddWall, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::KillOrganism:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryKillOrganism, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::ChooseOrganism:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TrySelectOrganism, pt.x + x, pt.y + y);
//                        goto endfor;
                        break;
                    case CursorMode::PlaceOrganism:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryAddOrganism, pt.x, pt.y);
                        goto endfor;
                    case CursorMode::DebugDisplayInfo:
                        edc.ui_user_actions_pool.emplace_back(ActionType::DebugDisplayInfo, pt.x, pt.y);
                    default:
                        break;
                }
            }
        }
    }
    endfor:
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ecp.do_not_use_user_actions_ui = false;

    last_last_cursor_x_pos = last_mouse_x_pos;
    last_last_cursor_y_pos = last_mouse_y_pos;
}

void MainWindow::change_main_grid_right_click() {
    ecp.do_not_use_user_actions_ui = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    while (ecp.do_not_use_user_actions_engine) {}

    if (update_last_cursor_pos) {
        last_last_cursor_x_pos = last_mouse_x_pos;
        last_last_cursor_y_pos = last_mouse_y_pos;

        update_last_cursor_pos = false;
    }

    auto cpg = calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    auto cpg2 = calculate_cursor_pos_on_grid(last_last_cursor_x_pos, last_last_cursor_y_pos);

    std::vector<Vector2<int>> points;

    points = iterate_between_two_points(cpg, cpg2);

    for (auto & pt: points) {
        for (int x = -brush_size / 2; x < float(brush_size) / 2; x++) {
            for (int y = -brush_size / 2; y < float(brush_size) / 2; y++) {
                switch (cursor_mode) {
                    case CursorMode::ModifyFood:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryRemoveFood, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::ModifyWall:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryRemoveWall, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::KillOrganism:
                        edc.ui_user_actions_pool.emplace_back(ActionType::TryKillOrganism, pt.x + x, pt.y + y);
                        break;
                    case CursorMode::ChooseOrganism:
                    case CursorMode::PlaceOrganism:
                    default:
                        break;
                }
            }
        }
    }
    ecp.do_not_use_user_actions_ui = false;

    last_last_cursor_x_pos = last_mouse_x_pos;
    last_last_cursor_y_pos = last_mouse_y_pos;
}

void MainWindow::change_editing_grid_left_click() {
    if (ee.change_disabled) { return;}

    auto cpg = ee.calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    if (cpg.x < 0 || cpg.y < 0 || cpg.x >= ee.editor_width || cpg.y >= ee.editor_height) { return;}

    //relative position
    auto r_pos = Vector2<int>{cpg.x - ee.editor_organism.x, cpg.y - ee.editor_organism.y};
    ee.editor_organism.anatomy.set_block(ee.chosen_block_type, ee.chosen_block_rotation, r_pos.x, r_pos.y);
    ee.create_image();
}

void MainWindow::change_editing_grid_right_click() {
    if (ee.change_disabled) { return;}

    auto cpg = ee.calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    if (cpg.x < 0 || cpg.y < 0 || cpg.x >= ee.editor_width || cpg.y >= ee.editor_height) { return;}
    if (ee.editor_organism.anatomy.organism_blocks.size() == 1) { return;}

    //relative position
    auto r_pos = Vector2<int>{cpg.x - ee.editor_organism.x, cpg.y - ee.editor_organism.y};
    ee.editor_organism.anatomy.set_block(BlockTypes::EmptyBlock, Rotation::UP, r_pos.x, r_pos.y);
    ee.create_image();
}

void MainWindow::load_textures_from_disk() {
    QImage image;
    auto executable_path = QCoreApplication::applicationDirPath().toStdString();

    std::array<std::string, 5> file_extensions{".png", ".jpg", ".jpeg", ".bmp", ".gif"};

    for (int i = 0; i < BLOCK_NAMES.size(); i++) {
        std::string filename;
        filename.append(executable_path).append("/textures/").append(
                [](std::string data){
                    std::transform(data.begin(), data.end(), data.begin(),
                                   [](unsigned char c){ return std::tolower(c); });
                    return data;
                }(BLOCK_NAMES[i])
                );

        bool exists = false;
        for (auto & extension: file_extensions) {
            if (std::filesystem::exists(filename + extension)) {
                filename.append(extension);
                exists = true;
                break;
            }
        }

        if (exists) {
            image.load(QString::fromStdString(filename));

            int width = image.width();
            int height = image.height();
            int total = width * height;

            textures.textures[i].width  = width;
            textures.textures[i].height = height;

            textures.textures[i].texture.resize(total);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    textures.textures[i].texture[x + y * width].r = qRed(image.pixel(x, y));
                    textures.textures[i].texture[x + y * width].g = qGreen(image.pixel(x, y));
                    textures.textures[i].texture[x + y * width].b = qBlue(image.pixel(x, y));
                }
            }
        } else {
            textures.textures[i] = Textures::default_holders[i];
        }
    }

#ifdef __CUDA_USED__
    if (cuda_is_available_var && use_cuda) {
        cuda_creator.copy_textures(textures);
        ee.cuda_image_creator.copy_textures(textures);
    }
#endif
}

void MainWindow::flip_fullscreen() {
    if (!is_fullscreen) {
        parentWidget()->showFullScreen();
        parentWidget()->show();
        is_fullscreen = true;
    } else {
        parentWidget()->showNormal();
        parentWidget()->show();
        is_fullscreen = false;
    }
    set_child_windows_always_on_top(is_fullscreen);
}

void MainWindow::set_child_windows_always_on_top(bool state) {
    ui.cb_statistics_always_on_top     ->setChecked(false);
    ui.cb_editor_always_on_top         ->setChecked(false);
    ui.cb_info_window_always_on_top    ->setChecked(false);
    ui.cb_recorder_window_always_on_top->setChecked(false);
    ui.cb_world_events_always_on_top   ->setChecked(false);
    ui.cb_benchmarks_always_on_top     ->setChecked(false);
    ui.cb_occp_always_on_top           ->setChecked(false);

    st.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    ee.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    iw.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    #ifndef __NO_RECORDER__
    rc.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    #endif
    we.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    bs.setWindowFlag(Qt::WindowStaysOnTopHint, state);
    occpw.setWindowFlag(Qt::WindowStaysOnTopHint, state);
}

void MainWindow::apply_font_to_windows(const QFont &_font) {
    setFont(_font);
    ee.setFont(_font);
    st.setFont(_font);
    iw.setFont(_font);
    #ifndef __NO_RECORDER__
    rc.setFont(_font);
    #endif
    we.setFont(_font);
    bs.setFont(_font);
    occpw.setFont(_font);
}

void MainWindow::save_state() {
    auto json_state_path = QCoreApplication::applicationDirPath().toStdString() + "/settings.json";
    DataSavingFunctions::write_json_state(json_state_path,
                                          DataSavingFunctions::ProgramState{
                                                  scaling_coefficient, keyboard_movement_amount,
                                                  SHIFT_keyboard_movement_multiplier,
                                                  font_size, float_precision, brush_size,
                                                  update_info_every_n_milliseconds,
                                                  use_cuda, disable_warnings,
                                                  really_stop_render, save_simulation_settings, uses_point_size
                                          }, sp, occp);
}

void MainWindow::load_state() {
    auto json_state_path = QCoreApplication::applicationDirPath().toStdString() + "/settings.json";
    DataSavingFunctions::read_json_state(json_state_path,
                                         DataSavingFunctions::ProgramState{
                                                 scaling_coefficient, keyboard_movement_amount,
                                                 SHIFT_keyboard_movement_multiplier,
                                                 font_size, float_precision, brush_size,
                                                 update_info_every_n_milliseconds,
                                                 use_cuda, disable_warnings,
                                                 really_stop_render, save_simulation_settings, uses_point_size
                                         }, sp, occp);
    initialize_gui();
    auto temp = disable_warnings;
    disable_warnings = true;
    cb_use_occ_slot(sp.use_occ);
    disable_warnings = temp;

    occpw.reinit_gui(true);

    apply_font_size();
}

void MainWindow::apply_font_size() {
    auto _font = font();

    if (uses_point_size) {
        _font.setPointSize(font_size);
    } else {
        _font.setPixelSize(font_size);
    }

    apply_font_to_windows(_font);
}

void MainWindow::create_image_creation_thread() {
    image_creation_thread = std::thread{[&](){
        auto point1 = std::chrono::high_resolution_clock::now();
        auto point2 = point1;
        while (ecp.make_images) {
            point1 = std::chrono::high_resolution_clock::now();
            if (!pause_grid_parsing || !really_stop_render) {
                if (have_read_buffer) {
                    have_read_buffer = false;
                    create_image();
                    image_frames++;
                }
            }
            point2 = std::chrono::high_resolution_clock::now();
            std::this_thread::sleep_for(std::chrono::microseconds(
                    std::max<int64_t>(
                            int(image_creation_interval * 1000000) -
                            std::chrono::duration_cast<std::chrono::microseconds>(point2 - point1).count()
            , 0)));
        }
    }};

    image_creation_thread.detach();
}

//https://gist.github.com/DavidMcLaughlin208/60e69e698e3858617c322d80a8f174e2
std::vector<Vector2<int>> MainWindow::iterate_between_two_points(Vector2<int> pos1, Vector2<int> pos2) {
    if (pos1.x == pos2.x && pos1.y == pos2.y) {return {pos1};}

    std::vector<Vector2<int>> points;
    int matrixX1 = pos1.x;
    int matrixY1 = pos1.y;
    int matrixX2 = pos2.x;
    int matrixY2 = pos2.y;

    int x_diff = matrixX1 - matrixX2;
    int y_diff = matrixY1 - matrixY2;
    bool x_diff_is_larger = std::abs(x_diff) > std::abs(y_diff);

    int x_modifier = x_diff < 0 ? 1 : -1;
    int y_modifier = y_diff < 0 ? 1 : -1;

    int longer_side_length  = std::max(std::abs(x_diff), std::abs(y_diff));
    int shorter_side_length = std::min(std::abs(x_diff), std::abs(y_diff));

    float slope = (shorter_side_length == 0 || longer_side_length == 0) ? 0 : ((float) (shorter_side_length) / (longer_side_length));

    int shorter_side_increase;
    for (int i = 1; i <= longer_side_length; i++) {
        shorter_side_increase = std::round(i * slope);
        int yIncrease, xIncrease;
        if (x_diff_is_larger) {
            xIncrease = i;
            yIncrease = shorter_side_increase;
        } else {
            yIncrease = i;
            xIncrease = shorter_side_increase;
        }
        int currentY = matrixY1 + (yIncrease * y_modifier);
        int currentX = matrixX1 + (xIncrease * x_modifier);
        points.emplace_back(currentX, currentY);
    }

    return points;
}