// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.03.2022.
//

#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
        QWidget(parent){
    _ui.setupUi(this);

    _ui.simulation_graphicsView->show();
    //If not false, the graphics view allows scrolling of an image after window resizing and only this helps.
    //Disabling graphics view doesn't change anything anyway.
    _ui.simulation_graphicsView->setEnabled(false);

    //https://stackoverflow.com/questions/32714105/mousemoveevent-is-not-called
    QCoreApplication::instance()->installEventFilter(this);

    s.init(&_ui);
    ee.init(15, 15, &_ui, &cc, &sp, &bp, &cursor_mode, &edc.chosen_organism);

    edc.simulation_width = 200;
    edc.simulation_height = 200;

#if __VALGRIND_MODE__
    edc.simulation_width = 50;
    edc.simulation_height = 50;
#endif

    edc.CPU_simulation_grid   .resize(edc.simulation_width, std::vector<AtomicGridBlock>(edc.simulation_height, AtomicGridBlock{}));
    edc.second_simulation_grid.resize(edc.simulation_width * edc.simulation_height, BaseGridBlock{});

    update_simulation_size_label();

    engine = new SimulationEngine(std::ref(edc), std::ref(ecp), std::ref(bp), std::ref(sp));

    engine->make_walls();

    //In mingw compiler std::random_device is deterministic?
    //https://stackoverflow.com/questions/18880654/why-do-i-get-the-same-sequence-for-every-run-with-stdrandom-device-with-mingw
//    boost::random_device rd;
//    std::seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
//    gen = lehmer64(rd());

    cc = ColorContainer{};
    sp = SimulationParameters{};

    auto anatomy = std::make_shared<Anatomy>();
    anatomy->set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    anatomy->set_block(BlockTypes::ProducerBlock, Rotation::UP, -1, -1);
    anatomy->set_block(BlockTypes::ProducerBlock, Rotation::UP, 1, 1);

//    anatomy->set_block(BlockTypes::EyeBlock, Rotation::UP, 0, 0);
//    anatomy->set_block(BlockTypes::MouthBlock, Rotation::UP, 1, 2);
//    anatomy->set_block(BlockTypes::KillerBlock, Rotation::UP, 1, 1);
//    anatomy->set_block(BlockTypes::ArmorBlock, Rotation::UP, -1, 1);
//    anatomy->set_block(BlockTypes::ProducerBlock, Rotation::UP, -1, 2);
//    anatomy->set_block(BlockTypes::MoverBlock, Rotation::UP, 0, 3);

    auto brain = std::make_shared<Brain>(BrainTypes::SimpleBrain);

    edc.base_organism = new Organism(edc.simulation_width / 2, edc.simulation_height / 2,
                                     Rotation::UP, anatomy, brain, &sp, &bp, 1);
    edc.chosen_organism = new Organism(edc.simulation_width / 2, edc.simulation_height / 2,
                                       Rotation::UP, std::make_shared<Anatomy>(anatomy), std::make_shared<Brain>(brain),
                                       &sp, &bp, 1);

    edc.base_organism->last_decision = DecisionObservation{};
    edc.chosen_organism->last_decision = DecisionObservation{};

    edc.to_place_organisms.push_back(new Organism(edc.base_organism));

    resize_image();
    reset_scale_view();

    //Will execute on first QT show event
    QTimer::singleShot(0, [&]{
        engine_thread = std::thread{&SimulationEngine::threaded_mainloop, engine};
        engine_thread.detach();

        fps_timer = clock_now();

        scene.addItem(&pixmap_item);
        _ui.simulation_graphicsView->setScene(&scene);

        reset_scale_view();
        initialize_gui();
        #if defined(__WIN32)
        ShowWindow(GetConsoleWindow(), SW_HIDE);
        #endif
    });

    timer = new QTimer(parent);
    //Will execute as fast as possible
    connect(timer, &QTimer::timeout, [&]{mainloop_tick();});

    set_window_interval(60);
    set_simulation_interval(60);

    timer->start();
#if __VALGRIND_MODE__ == 1
    cb_synchronise_simulation_and_window_slot(false);
    _ui.cb_synchronise_sim_and_win->setChecked(false);
#endif
}

void MainWindow::mainloop_tick() {
    if (synchronise_simulation_and_window) {
        ecp.engine_pass_tick = true;
        ecp.synchronise_simulation_tick = true;
    }
    if (ecp.update_editor_organism) { ee.load_chosen_organism(), ecp.update_editor_organism = false;}

    window_tick();
    window_frames++;

    auto info_update = std::chrono::duration_cast<std::chrono::milliseconds>(clock_now() - fps_timer).count();

    if (synchronise_info_update_with_window_update || info_update > update_info_every_n_milliseconds) {
        auto start_timer = clock_now();
        pause_engine();
        uint32_t simulation_frames = edc.engine_ticks;
        edc.engine_ticks = 0;

        if (info_update == 0) {info_update = 1;}

        auto scale = (info_update/1000.);

        auto info = parse_organisms_info();

        unpause_engine();

        update_fps_labels(window_frames/scale, simulation_frames/scale);
        update_statistics_info(info);

        window_frames = 0;
        fps_timer = clock_now();
    }
}

void MainWindow::update_fps_labels(int fps, int sps) {
    _ui.lb_fps->setText(QString::fromStdString("fps: " + std::to_string(fps)));
    _ui.lb_sps->setText(QString::fromStdString("sps: "+std::to_string(sps)));
}

void MainWindow::window_tick() {
    if (resize_simulation_grid_flag) { resize_simulation_grid(); resize_simulation_grid_flag=false;}
    if (ecp.tb_paused) {_ui.tb_pause->setChecked(true);}

    if (left_mouse_button_pressed  && change_main_simulation_grid) {change_main_grid_left_click();}
    if (right_mouse_button_pressed && change_main_simulation_grid) {change_main_grid_right_click();}

    if (left_mouse_button_pressed  && change_editing_grid) {change_editing_grid_left_click();}
    if (right_mouse_button_pressed && change_editing_grid) {change_editing_grid_right_click();}

#if __VALGRIND_MODE__ == 1
    return;
#endif
    if (pause_grid_parsing && really_stop_render) { return;}
    create_image();
}

void MainWindow::resize_image() {
    image_vector.clear();
    image_vector.reserve(4 * _ui.simulation_graphicsView->viewport()->width() * _ui.simulation_graphicsView->viewport()->height());
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
        if (_ui.simulation_graphicsView->viewport()->height() < _ui.simulation_graphicsView->viewport()->width()) {
            exp = log((float) edc.simulation_height / (float) _ui.simulation_graphicsView->viewport()->height()) /
                  log(scaling_coefficient);
        } else {
            exp = log((float) edc.simulation_width / (float) _ui.simulation_graphicsView->viewport()->width()) /
                  log(scaling_coefficient);
        }
    // if not equal, then to capture full view you need to scale by largest dimension
    } else {
        if (edc.simulation_width > edc.simulation_height) {
            exp = log((float) edc.simulation_width / (float) _ui.simulation_graphicsView->viewport()->width()) /
                  log(scaling_coefficient);
        } else {
            exp = log((float) edc.simulation_height / (float) _ui.simulation_graphicsView->viewport()->height()) /
                  log(scaling_coefficient);
        }
    }
    scaling_zoom = pow(scaling_coefficient, exp);
}

color &MainWindow::get_color_simplified(BlockTypes type) {
    switch (type) {
        case BlockTypes::EmptyBlock :   return cc.empty_block;
        case BlockTypes::MouthBlock:    return cc.mouth;
        case BlockTypes::ProducerBlock: return cc.producer;
        case BlockTypes::MoverBlock:    return cc.mover;
        case BlockTypes::KillerBlock:   return cc.killer;
        case BlockTypes::ArmorBlock:    return cc.armor;
        case BlockTypes::EyeBlock:      return cc.eye;
        case BlockTypes::FoodBlock:     return cc.food;
        case BlockTypes::WallBlock:     return cc.wall;
        default: return cc.empty_block;
    }
}

color & MainWindow::get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale) {
    int x;
    int y;

    switch (type) {
        case BlockTypes::EmptyBlock :   return cc.empty_block;
        case BlockTypes::MouthBlock:    return cc.mouth;
        case BlockTypes::ProducerBlock: return cc.producer;
        case BlockTypes::MoverBlock:    return cc.mover;
        case BlockTypes::KillerBlock:   return cc.killer;
        case BlockTypes::ArmorBlock:    return cc.armor;
        case BlockTypes::EyeBlock:
            x = relative_x_scale * 5;
            y = relative_y_scale * 5;
            {
                switch (rotation) {
                    case Rotation::UP:
                        break;
                    case Rotation::LEFT:
                        x -= 2;
                        y -= 2;
                        std::swap(x, y);
                        x = -x;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::DOWN:
                        x -= 2;
                        y -= 2;
                        x = -x;
                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::RIGHT:
                        x -= 2;
                        y -= 2;
                        std::swap(x, y);
                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                }
            }
            return textures.rawEyeTexture[x + y * 5];
        case BlockTypes::FoodBlock:     return cc.food;
        case BlockTypes::WallBlock:     return cc.wall;
        default: return cc.empty_block;
    }
}

void MainWindow::create_image() {
    resize_image();
    auto image_width  = _ui.simulation_graphicsView->viewport()->width();
    auto image_height = _ui.simulation_graphicsView->viewport()->height();

    int scaled_width  = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and y coordinates on simulation grid
    auto start_x = int(center_x-(scaled_width / 2));
    auto end_x   = int(center_x+(scaled_width / 2));

    auto start_y = int(center_y-(scaled_height / 2));
    auto end_y   = int(center_y+(scaled_height / 2));

    std::vector<int> lin_width;
    std::vector<int> lin_height;

    calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    std::vector<int> truncated_lin_width;  truncated_lin_width .reserve(image_width);
    std::vector<int> truncated_lin_height; truncated_lin_height.reserve(image_height);

    calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);

    if ((!pause_grid_parsing && !ecp.engine_global_pause) || synchronise_simulation_and_window) {
        ecp.engine_pause = true;
        // pausing engine to parse data from engine.
        auto paused = wait_for_engine_to_pause();
        // if for some reason engine is not paused in time, it will use old parsed data and not switch engine on.
        if (paused) {parse_simulation_grid(truncated_lin_width, truncated_lin_height); unpause_engine();}
    }
    if (simplified_rendering) {
        simplified_image_creation(image_width, image_height, lin_width, lin_height);
    } else {
        if (!use_cuda) {
            complex_image_creation(lin_width, lin_height);
        } else {
#if __CUDA_USED__
            cuda_creator.cuda_create_image(image_width,
                                           image_height,
                                           lin_width,
                                           lin_height,
                                           image_vector,
                                           cc,
                                           edc, 32, truncated_lin_width, truncated_lin_height);
#endif
        }
    }
    pixmap_item.setPixmap(QPixmap::fromImage(QImage(image_vector.data(), image_width, image_height, QImage::Format_RGB32)));
}

void MainWindow::calculate_linspace(std::vector<int> & lin_width, std::vector<int> & lin_height,
                                    int start_x, int end_x, int start_y, int end_y, int image_width, int image_height) {
    lin_width  = linspace<int>(start_x, end_x, image_width);
    lin_height = linspace<int>(start_y, end_y, image_height);

    //when zoomed, boundaries of simulation grid are more than could be displayed by 1, so we need to delete the last
    // n pixels
    int max_x = lin_width[lin_width.size()-1];
    int max_y = lin_height[lin_height.size()-1];
    int del_x = 0;
    int del_y = 0;
    for (int x = lin_width.size() -1; lin_width[x]  == max_x; x--) {del_x++;}
    for (int y = lin_height.size()-1; lin_height[y] == max_y; y--) {del_y++;}

    for (int i = 0; i < del_x; i++) {lin_width.pop_back();}
    for (int i = 0; i < del_y; i++) {lin_height.pop_back();}
}

void MainWindow::calculate_truncated_linspace(
        int image_width, int image_height,
        const std::vector<int> &lin_width,
        const std::vector<int> &lin_height,
        std::vector<int> & truncated_lin_width,
        std::vector<int> & truncated_lin_height) {

    int min_val = INT32_MIN;
    for (int x = 0; x < image_width; x++) {if (lin_width[x] > min_val) {min_val = lin_width[x]; truncated_lin_width.push_back(min_val);}}
    truncated_lin_width.pop_back();
    min_val = INT32_MIN;
    for (int y = 0; y < image_height; y++) {if (lin_height[y] > min_val) {min_val = lin_height[y]; truncated_lin_height.push_back(min_val);}}
    truncated_lin_height.pop_back();
}

void MainWindow::simplified_image_creation(int image_width, int image_height,
                                           const std::vector<int> &lin_width,
                                           const std::vector<int> &lin_height) {
    color pixel_color;
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            if (lin_width[x] < 0 || lin_width[x] >= edc.simulation_width || lin_height[y] < 0 || lin_height[y] >= edc.simulation_height) { pixel_color = cc.simulation_background_color;}
            else {pixel_color = get_color_simplified(edc.second_simulation_grid[lin_width[x] + lin_height[y] * edc.simulation_width].type);}
            set_image_pixel(x, y, pixel_color);
        }
    }
}

void MainWindow::complex_image_creation(const std::vector<int> &lin_width, const std::vector<int> &lin_height) {
    //x - start, y - stop
    std::vector<Vector2<int>> width_img_boundaries;
    std::vector<Vector2<int>> height_img_boundaries;

    auto last = INT32_MIN;
    auto count = 0;
    for (int x = 0; x < lin_width.size(); x++) {
        if (last < lin_width[x]) {
            width_img_boundaries.emplace_back(count, x);
            last = lin_width[x];
            count = x;
        }
    }
    width_img_boundaries.emplace_back(count, lin_width.size());

    last = INT32_MIN;
    count = 0;
    for (int x = 0; x < lin_height.size(); x++) {
        if (last < lin_height[x]) {
            height_img_boundaries.emplace_back(count, x);
            last = lin_height[x];
            count = x;
        }
    }
    height_img_boundaries.emplace_back(count, lin_height.size());

    color pixel_color;
    //width of boundaries of an organisms

    //width bound, height bound
    for (auto &w_b: width_img_boundaries) {
        for (auto &h_b: height_img_boundaries) {
            for (int x = w_b.x; x < w_b.y; x++) {
                for (int y = h_b.x; y < h_b.y; y++) {
                    auto &block = edc.second_simulation_grid[lin_width[x] + lin_height[y] * edc.simulation_width];

                    if (lin_width[x] < 0 ||
                        lin_width[x] >= edc.simulation_width ||
                        lin_height[y] < 0 ||
                        lin_height[y] >= edc.simulation_height) {
                        pixel_color = cc.simulation_background_color;
                    } else {
                        pixel_color = get_texture_color(block.type,
                                                        block.rotation,
                                                        float(x - w_b.x) / (w_b.y - w_b.x),
                                                        float(y - h_b.x) / (h_b.y - h_b.x));
                    }
                    set_image_pixel(x, y, pixel_color);
                }
            }
        }
    }
}

// depth * ( y * width + x) + z
// depth * width * y + depth * x + z
void MainWindow::set_image_pixel(int x, int y, const color &color) {
    auto index = 4 * (y * _ui.simulation_graphicsView->viewport()->width() + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

void MainWindow::set_window_interval(int max_window_fps) {
    if (max_window_fps <= 0) {
        window_interval = 0.;
        timer->setInterval(0);
        return;
    }
    window_interval = 1./max_window_fps;
    timer->setInterval(1000/max_window_fps);
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


//Can wait, or it can not.
bool MainWindow::wait_for_engine_to_pause() {
    if (!wait_for_engine_to_stop_to_render || ecp.engine_global_pause) {return true;}
    return wait_for_engine_to_pause_force();
}

//Will always wait for engine to pause
bool MainWindow::wait_for_engine_to_pause_force() {
    while (!ecp.engine_paused) {}
    return ecp.engine_paused;
}

bool MainWindow::wait_for_engine_to_pause_processing_user_actions() {
    while (ecp.processing_user_actions) {}
    return !ecp.processing_user_actions;
}

//TODO rare crashes could be because of parsing. Needs further research.
void MainWindow::parse_simulation_grid(const std::vector<int> &lin_width, const std::vector<int> &lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= edc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= edc.simulation_height) { continue; }
            edc.second_simulation_grid[x + y * edc.simulation_width].type = edc.CPU_simulation_grid[x][y].type;
            edc.second_simulation_grid[x + y * edc.simulation_width].rotation = edc.CPU_simulation_grid[x][y].rotation;
        }
    }
}

void MainWindow::parse_full_simulation_grid(bool parse) {
    if (!parse) {return;}
    pause_engine();

    for (int x = 0; x < edc.simulation_width; x++) {
        for (int y = 0; y < edc.simulation_height; y++) {
            edc.second_simulation_grid[x + y * edc.simulation_width].type = edc.CPU_simulation_grid[x][y].type;
            edc.second_simulation_grid[x + y * edc.simulation_width].rotation = edc.CPU_simulation_grid[x][y].rotation;
        }
    }

    unpause_engine();
}

void MainWindow::set_simulation_num_threads(uint8_t num_threads) {
    pause_engine();

    ecp.num_threads = num_threads;
    if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        ecp.build_threads = true;
    }

    unpause_engine();
}

void MainWindow::set_cursor_mode(CursorMode mode) {
    cursor_mode = mode;
}

void MainWindow::set_simulation_mode(SimulationModes mode) {
    ecp.change_to_mode = mode;
    ecp.change_simulation_mode = true;
}

void MainWindow::calculate_new_simulation_size() {
    auto window_size = _ui.simulation_graphicsView->viewport()->size();

    new_simulation_width  = window_size.width() / starting_cell_size_on_resize;
    new_simulation_height = window_size.height() / starting_cell_size_on_resize;
}

void MainWindow::resize_simulation_grid() {
    if (fill_window) {calculate_new_simulation_size();}

    if (!disable_warnings) {
        if (!use_cuda) {
            auto msg = DescisionMessageBox("Warning",
                                       QString::fromStdString("Simulation space will be rebuilt and all organisms cleared.\n"
                                       "New grid will need " + convert_num_bytes((sizeof(BaseGridBlock) + sizeof(AtomicGridBlock))*new_simulation_height*new_simulation_width)),
                                       "OK", "Cancel", this);
            auto result = msg.exec();
            if (!result) {
                return;
            }
        } else {
            auto msg = DescisionMessageBox("Warning",
                                           QString::fromStdString("Simulation space will be rebuilt and all organisms cleared.\n"
                                                                  "New grid will need " + convert_num_bytes((sizeof(BaseGridBlock) + sizeof(AtomicGridBlock))*new_simulation_height*new_simulation_width)
                                                                  + " of RAM and " + convert_num_bytes(sizeof(BaseGridBlock)*new_simulation_height*new_simulation_width))
                                                                  + " GPU's VRAM",
                                           "OK", "Cancel", this);
            auto result = msg.exec();
            if (!result) {
                return;
            }
        }
    }

    pause_engine();

    edc.simulation_width = new_simulation_width;
    edc.simulation_height = new_simulation_height;

    edc.CPU_simulation_grid.clear();
    edc.second_simulation_grid.clear();

    edc.CPU_simulation_grid   .resize(edc.simulation_width, std::vector<AtomicGridBlock>(edc.simulation_height, AtomicGridBlock{}));
    edc.second_simulation_grid.resize(edc.simulation_width * edc.simulation_height, BaseGridBlock{});

    if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
        ecp.build_threads = true;
    }

    engine->init_auto_food_drop(edc.simulation_width, edc.simulation_height);

    engine->reset_world();
    unpause_engine();

    reset_scale_view();
}

void MainWindow::clear_world() {
    engine->partial_clear_world();
    engine->make_walls();
    unpause_engine();
}

void MainWindow::pause_engine() {
    ecp.engine_pause = true;
    wait_for_engine_to_pause_force();
}

void MainWindow::unpause_engine() {
    if (!synchronise_simulation_and_window) {
        ecp.engine_pause = false;
    }
}

OrganismAvgBlockInformation MainWindow::parse_organisms_info() {
    OrganismAvgBlockInformation info;

    bool has_pool = true;
    int i = 0;
    //Why while loop? the easier implementation with for loop randomly crashes sometimes, and I don't know why.
    while (has_pool) {
        std::vector<Organism*> * pool;

        if (ecp.simulation_mode == SimulationModes::CPU_Single_Threaded) {
            pool = &edc.organisms;
            has_pool = false;
        } else if (ecp.simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
            pool = &edc.organisms_pools[i];
            i++;
            if (i >= ecp.num_threads) {
                has_pool = false;
            }
        } else {
            throw "no pool";
        }

        for (auto & organism: *pool) {
            info.total_size_organism_blocks += organism->anatomy->_organism_blocks.size();
            info.total_size_producing_space += organism->anatomy->_producing_space.size();
            info.total_size_eating_space    += organism->anatomy->_eating_space.size();

            if (organism->anatomy->_mover_blocks > 0) {
                info.move_range += organism->move_range;
                info.moving_organisms++;

                if (organism->anatomy->_eye_blocks > 0) {
                    info.organisms_with_eyes++;
                }
            }

            info.total_avg.size += organism->anatomy->_organism_blocks.size();

            info.total_avg._organism_lifetime += organism->max_lifetime;
            info.total_avg._organism_age      += organism->lifetime;
            info.total_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
            info.total_avg._producer_blocks   += organism->anatomy->_producer_blocks;
            info.total_avg._mover_blocks      += organism->anatomy->_mover_blocks;
            info.total_avg._killer_blocks     += organism->anatomy->_killer_blocks;
            info.total_avg._armor_blocks      += organism->anatomy->_armor_blocks;
            info.total_avg._eye_blocks        += organism->anatomy->_eye_blocks;

            info.total_avg.brain_mutation_rate   += organism->brain_mutation_rate;
            info.total_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
            info.total_avg.total++;

            if (organism->anatomy->_mover_blocks > 0) {
                info.moving_avg.size += organism->anatomy->_organism_blocks.size();

                info.moving_avg._organism_lifetime += organism->max_lifetime;
                info.moving_avg._organism_age      += organism->lifetime;
                info.moving_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
                info.moving_avg._producer_blocks   += organism->anatomy->_producer_blocks;
                info.moving_avg._mover_blocks      += organism->anatomy->_mover_blocks;
                info.moving_avg._killer_blocks     += organism->anatomy->_killer_blocks;
                info.moving_avg._armor_blocks      += organism->anatomy->_armor_blocks;
                info.moving_avg._eye_blocks        += organism->anatomy->_eye_blocks;

                info.moving_avg.brain_mutation_rate   += organism->brain_mutation_rate;
                info.moving_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
                info.moving_avg.total++;
            } else {
                info.station_avg.size += organism->anatomy->_organism_blocks.size();

                info.station_avg._organism_lifetime += organism->max_lifetime;
                info.station_avg._organism_age      += organism->lifetime;
                info.station_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
                info.station_avg._producer_blocks   += organism->anatomy->_producer_blocks;
                info.station_avg._mover_blocks      += organism->anatomy->_mover_blocks;
                info.station_avg._killer_blocks     += organism->anatomy->_killer_blocks;
                info.station_avg._armor_blocks      += organism->anatomy->_armor_blocks;
                info.station_avg._eye_blocks        += organism->anatomy->_eye_blocks;

                info.station_avg.brain_mutation_rate   += organism->brain_mutation_rate;
                info.station_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
                info.station_avg.total++;
            }
        }
    }

    info.total_size_organism_blocks                *= sizeof(SerializedOrganismBlockContainer);
    info.total_size_producing_space                *= sizeof(SerializedAdjacentSpaceContainer);
    info.total_size_eating_space                   *= sizeof(SerializedAdjacentSpaceContainer);
    info.total_size_single_adjacent_space          *= sizeof(SerializedAdjacentSpaceContainer);
    info.total_size_single_diagonal_adjacent_space *= sizeof(SerializedAdjacentSpaceContainer);

    info.move_range /= info.moving_organisms;

    info.total_size = info.total_size_organism_blocks +
                      info.total_size_producing_space +
                      info.total_size_eating_space +
                      info.total_size_single_adjacent_space +
                      info.total_size_single_diagonal_adjacent_space +
                      (sizeof(Brain) * info.total_avg.total) +
                      (sizeof(Anatomy) * info.total_avg.total) +
                      (sizeof(Organism) * info.total_avg.total)
                      ;

    info.total_total_mutation_rate = info.total_avg.anatomy_mutation_rate;

    info.total_avg.size /= info.total_avg.total;

    info.total_avg._organism_lifetime /= info.total_avg.total;
    info.total_avg._organism_age      /= info.total_avg.total;
    info.total_avg._mouth_blocks      /= info.total_avg.total;
    info.total_avg._producer_blocks   /= info.total_avg.total;
    info.total_avg._mover_blocks      /= info.total_avg.total;
    info.total_avg._killer_blocks     /= info.total_avg.total;
    info.total_avg._armor_blocks      /= info.total_avg.total;
    info.total_avg._eye_blocks        /= info.total_avg.total;

    info.total_avg.brain_mutation_rate   /= info.total_avg.total;
    info.total_avg.anatomy_mutation_rate /= info.total_avg.total;

    if (std::isnan(info.total_avg.size))             {info.total_avg.size = 0;}
    if (std::isnan(info.move_range))                 {info.move_range     = 0;}

    if (std::isnan(info.total_avg._organism_lifetime)) {info.total_avg._organism_lifetime = 0;}
    if (std::isnan(info.total_avg._organism_age))      {info.total_avg._organism_age      = 0;}
    if (std::isnan(info.total_avg._mouth_blocks))      {info.total_avg._mouth_blocks      = 0;}
    if (std::isnan(info.total_avg._producer_blocks))   {info.total_avg._producer_blocks   = 0;}
    if (std::isnan(info.total_avg._mover_blocks))      {info.total_avg._mover_blocks      = 0;}
    if (std::isnan(info.total_avg._killer_blocks))     {info.total_avg._killer_blocks     = 0;}
    if (std::isnan(info.total_avg._armor_blocks))      {info.total_avg._armor_blocks      = 0;}
    if (std::isnan(info.total_avg._eye_blocks))        {info.total_avg._eye_blocks        = 0;}

    if (std::isnan(info.total_avg.brain_mutation_rate))   {info.total_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.total_avg.anatomy_mutation_rate)) {info.total_avg.anatomy_mutation_rate = 0;}


    info.moving_avg.size /= info.moving_avg.total;

    info.moving_avg._organism_lifetime /= info.moving_avg.total;
    info.moving_avg._organism_age      /= info.moving_avg.total;
    info.moving_avg._mouth_blocks      /= info.moving_avg.total;
    info.moving_avg._producer_blocks   /= info.moving_avg.total;
    info.moving_avg._mover_blocks      /= info.moving_avg.total;
    info.moving_avg._killer_blocks     /= info.moving_avg.total;
    info.moving_avg._armor_blocks      /= info.moving_avg.total;
    info.moving_avg._eye_blocks        /= info.moving_avg.total;

    info.moving_avg.brain_mutation_rate   /= info.moving_avg.total;
    info.moving_avg.anatomy_mutation_rate /= info.moving_avg.total;

    if (std::isnan(info.moving_avg.size))             {info.moving_avg.size             = 0;}

    if (std::isnan(info.moving_avg._organism_lifetime)) {info.moving_avg._organism_lifetime = 0;}
    if (std::isnan(info.moving_avg._organism_age))      {info.moving_avg._organism_age      = 0;}
    if (std::isnan(info.moving_avg._mouth_blocks))      {info.moving_avg._mouth_blocks      = 0;}
    if (std::isnan(info.moving_avg._producer_blocks))   {info.moving_avg._producer_blocks   = 0;}
    if (std::isnan(info.moving_avg._mover_blocks))      {info.moving_avg._mover_blocks      = 0;}
    if (std::isnan(info.moving_avg._killer_blocks))     {info.moving_avg._killer_blocks     = 0;}
    if (std::isnan(info.moving_avg._armor_blocks))      {info.moving_avg._armor_blocks      = 0;}
    if (std::isnan(info.moving_avg._eye_blocks))        {info.moving_avg._eye_blocks        = 0;}

    if (std::isnan(info.moving_avg.brain_mutation_rate))   {info.moving_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.moving_avg.anatomy_mutation_rate)) {info.moving_avg.anatomy_mutation_rate = 0;}


    info.station_avg.size /= info.station_avg.total;

    info.station_avg._organism_lifetime /= info.station_avg.total;
    info.station_avg._organism_age      /= info.station_avg.total;
    info.station_avg._mouth_blocks      /= info.station_avg.total;
    info.station_avg._producer_blocks   /= info.station_avg.total;
    info.station_avg._mover_blocks      /= info.station_avg.total;
    info.station_avg._killer_blocks     /= info.station_avg.total;
    info.station_avg._armor_blocks      /= info.station_avg.total;
    info.station_avg._eye_blocks        /= info.station_avg.total;

    info.station_avg.brain_mutation_rate   /= info.station_avg.total;
    info.station_avg.anatomy_mutation_rate /= info.station_avg.total;

    if (std::isnan(info.station_avg.size))             {info.station_avg.size             = 0;}

    if (std::isnan(info.station_avg._organism_lifetime)) {info.station_avg._organism_lifetime = 0;}
    if (std::isnan(info.station_avg._organism_age))      {info.station_avg._organism_age      = 0;}
    if (std::isnan(info.station_avg._mouth_blocks))      {info.station_avg._mouth_blocks      = 0;}
    if (std::isnan(info.station_avg._producer_blocks))   {info.station_avg._producer_blocks   = 0;}
    if (std::isnan(info.station_avg._mover_blocks))      {info.station_avg._mover_blocks      = 0;}
    if (std::isnan(info.station_avg._killer_blocks))     {info.station_avg._killer_blocks     = 0;}
    if (std::isnan(info.station_avg._armor_blocks))      {info.station_avg._armor_blocks      = 0;}
    if (std::isnan(info.station_avg._eye_blocks))        {info.station_avg._eye_blocks        = 0;}

    if (std::isnan(info.station_avg.brain_mutation_rate))   {info.station_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.station_avg.anatomy_mutation_rate)) {info.station_avg.anatomy_mutation_rate = 0;}
    return info;
}

void MainWindow::update_statistics_info(const OrganismAvgBlockInformation &info) {
    s._ui.lb_total_engine_ticks ->setText(QString::fromStdString("Total engine ticks: "    + std::to_string(edc.total_engine_ticks)));
    s._ui.lb_organisms_memory_consumption->setText(QString::fromStdString("Organisms's memory consumption: " +
                                                                                convert_num_bytes(info.total_size)));
    s._ui.lb_organisms_alive_2    ->setText(QString::fromStdString("Organism alive: "        + std::to_string(info.total_avg.total)));
    s._ui.lb_organism_size_4      ->setText(QString::fromStdString("Avg organism size: "     + to_str(info.total_avg.size,               float_precision)));
    s._ui.lb_avg_org_lifetime_4   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.total_avg._organism_lifetime, float_precision)));
    s._ui.lb_avg_age_4            ->setText(QString::fromStdString("Avg organism age: "      + to_str(info.total_avg._organism_age,      float_precision)));
    s._ui.lb_mouth_num_4          ->setText(QString::fromStdString("Avg mouth num: "         + to_str(info.total_avg._mouth_blocks,      float_precision)));
    s._ui.lb_producer_num_4       ->setText(QString::fromStdString("Avg producer num: "      + to_str(info.total_avg._producer_blocks,   float_precision)));
    s._ui.lb_mover_num_4          ->setText(QString::fromStdString("Avg mover num: "         + to_str(info.total_avg._mover_blocks,      float_precision)));
    s._ui.lb_killer_num_4         ->setText(QString::fromStdString("Avg killer num: "        + to_str(info.total_avg._killer_blocks,     float_precision)));
    s._ui.lb_armor_num_4          ->setText(QString::fromStdString("Avg armor num: "         + to_str(info.total_avg._armor_blocks,      float_precision)));
    s._ui.lb_eye_num_4            ->setText(QString::fromStdString("Avg eye num: "           + to_str(info.total_avg._eye_blocks,        float_precision)));
    s._ui.lb_anatomy_mutation_rate_4 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.total_avg.anatomy_mutation_rate, float_precision)));
    s._ui.lb_brain_mutation_rate_4   ->setText(QString::fromStdString("Avg brain mutation rate: "   + to_str(info.total_avg.brain_mutation_rate,   float_precision)));


    s._ui.lb_moving_organisms     ->setText(QString::fromStdString("Moving organisms: "      + std::to_string(info.moving_avg.total)));
    s._ui.lb_organisms_with_eyes  ->setText(QString::fromStdString("Organisms with eyes: "   + std::to_string(info.organisms_with_eyes)));
    s._ui.lb_avg_org_lifetime_2   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.moving_avg._organism_lifetime, float_precision)));
    s._ui.lb_avg_age_2            ->setText(QString::fromStdString("Avg organism age: "      + to_str(info.moving_avg._organism_age,      float_precision)));
    s._ui.lb_average_moving_range ->setText(QString::fromStdString("Avg moving range: "      + to_str(info.move_range,                    float_precision)));
    s._ui.lb_organism_size_2      ->setText(QString::fromStdString("Avg organism size: "     + to_str(info.moving_avg.size,               float_precision)));
    s._ui.lb_mouth_num_2          ->setText(QString::fromStdString("Avg mouth num: "         + to_str(info.moving_avg._mouth_blocks,      float_precision)));
    s._ui.lb_producer_num_2       ->setText(QString::fromStdString("Avg producer num: "      + to_str(info.moving_avg._producer_blocks,   float_precision)));
    s._ui.lb_mover_num_2          ->setText(QString::fromStdString("Avg mover num: "         + to_str(info.moving_avg._mover_blocks,      float_precision)));
    s._ui.lb_killer_num_2         ->setText(QString::fromStdString("Avg killer num: "        + to_str(info.moving_avg._killer_blocks,     float_precision)));
    s._ui.lb_armor_num_2          ->setText(QString::fromStdString("Avg armor num: "         + to_str(info.moving_avg._armor_blocks,      float_precision)));
    s._ui.lb_eye_num_2            ->setText(QString::fromStdString("Avg eye num: "           + to_str(info.moving_avg._eye_blocks,        float_precision)));
    s._ui.lb_anatomy_mutation_rate_2 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.moving_avg.anatomy_mutation_rate, float_precision)));
    s._ui.lb_brain_mutation_rate_2   ->setText(QString::fromStdString("Avg brain mutation rate: "   + to_str(info.moving_avg.brain_mutation_rate,   float_precision)));


    s._ui.lb_stationary_organisms ->setText(QString::fromStdString("Stationary organisms: "  + std::to_string(info.station_avg.total)));
    s._ui.lb_organism_size_3      ->setText(QString::fromStdString("Avg organism size: "     + to_str(info.station_avg.size,               float_precision)));
    s._ui.lb_avg_org_lifetime_3   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.station_avg._organism_lifetime, float_precision)));
    s._ui.lb_avg_age_3            ->setText(QString::fromStdString("Avg organism age: "      + to_str(info.station_avg._organism_age,      float_precision)));
    s._ui.lb_mouth_num_3          ->setText(QString::fromStdString("Avg mouth num: "         + to_str(info.station_avg._mouth_blocks,      float_precision)));
    s._ui.lb_producer_num_3       ->setText(QString::fromStdString("Avg producer num: "      + to_str(info.station_avg._producer_blocks,   float_precision)));
    s._ui.lb_killer_num_3         ->setText(QString::fromStdString("Avg killer num: "        + to_str(info.station_avg._killer_blocks,     float_precision)));
    s._ui.lb_armor_num_3          ->setText(QString::fromStdString("Avg armor num: "         + to_str(info.station_avg._armor_blocks,      float_precision)));
    s._ui.lb_eye_num_3            ->setText(QString::fromStdString("Avg eye num: "           + to_str(info.station_avg._eye_blocks,        float_precision)));
    s._ui.lb_anatomy_mutation_rate_3 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.station_avg.anatomy_mutation_rate, float_precision)));
    s._ui.lb_brain_mutation_rate_3   ->setText(QString::fromStdString("Avg brain mutation rate: "   + to_str(info.station_avg.brain_mutation_rate,   float_precision)));
}

// So that changes in code values would be set by default in gui.
void MainWindow::initialize_gui() {
    //World settings
    _ui.le_cell_size         ->setText(QString::fromStdString(std::to_string(starting_cell_size_on_resize)));
    _ui.le_simulation_width  ->setText(QString::fromStdString(std::to_string(edc.simulation_width)));
    _ui.le_simulation_height ->setText(QString::fromStdString(std::to_string(edc.simulation_height)));
    _ui.le_max_organisms     ->setText(QString::fromStdString(std::to_string(edc.max_organisms)));
    _ui.le_brush_size        ->setText(QString::fromStdString(std::to_string(brush_size)));
    _ui.cb_reset_on_total_extinction ->setChecked(sp.reset_on_total_extinction);
    _ui.cb_pause_on_total_extinction ->setChecked(sp.pause_on_total_extinction);
    _ui.cb_fill_window               ->setChecked(fill_window);
    //Evolution settings
    _ui.le_food_production_probability       ->setText(QString::fromStdString(to_str(sp.food_production_probability,          4)));
    _ui.le_global_anatomy_mutation_rate      ->setText(QString::fromStdString(to_str(sp.global_anatomy_mutation_rate,         2)));
    _ui.le_global_brain_mutation_rate        ->setText(QString::fromStdString(to_str(sp.global_brain_mutation_rate,           2)));
    _ui.le_anatomy_mutation_rate_delimiter   ->setText(QString::fromStdString(to_str(sp.anatomy_mutation_rate_delimiter,      2)));
    _ui.le_brain_mutation_rate_delimiter     ->setText(QString::fromStdString(to_str(sp.brain_mutation_rate_delimiter,        2)));
    _ui.le_move_range_delimiter              ->setText(QString::fromStdString(to_str(sp.move_range_delimiter,                 2)));
    _ui.le_lifespan_multiplier               ->setText(QString::fromStdString(to_str(sp.lifespan_multiplier,                  3)));
    _ui.le_brain_min_possible_mutation_rate  ->setText(QString::fromStdString(to_str(sp.brain_min_possible_mutation_rate,     3)));
    _ui.le_anatomy_min_possible_mutation_rate->setText(QString::fromStdString(to_str(sp.anatomy_min_possible_mutation_rate,   3)));
    _ui.le_extra_mover_reproduction_cost     ->setText(QString::fromStdString(to_str(sp.extra_mover_reproductive_cost,        0)));
    _ui.le_extra_reproduction_cost           ->setText(QString::fromStdString(to_str(sp.extra_reproduction_cost,              0)));
    _ui.le_anatomy_mutation_rate_step        ->setText(QString::fromStdString(to_str(sp.anatomy_mutations_rate_mutation_step, 2)));
    _ui.le_brain_mutation_rate_step          ->setText(QString::fromStdString(to_str(sp.brain_mutation_rate_mutation_step,    2)));
    _ui.le_produce_food_every_n_tick         ->setText(QString::fromStdString(std::to_string(sp.produce_food_every_n_life_ticks)));
    _ui.le_look_range                        ->setText(QString::fromStdString(std::to_string(sp.look_range)));
    _ui.le_auto_produce_n_food               ->setText(QString::fromStdString(std::to_string(sp.auto_produce_n_food)));
    _ui.le_auto_produce_food_every_n_tick    ->setText(QString::fromStdString(std::to_string(sp.auto_produce_food_every_n_ticks)));
    _ui.le_add                               ->setText(QString::fromStdString(std::to_string(sp.add_cell)));
    _ui.le_change                            ->setText(QString::fromStdString(std::to_string(sp.change_cell)));
    _ui.le_remove                            ->setText(QString::fromStdString(std::to_string(sp.remove_cell)));
    _ui.le_min_reproduction_distance         ->setText(QString::fromStdString(std::to_string(sp.min_reproducing_distance)));
    _ui.le_max_reproduction_distance         ->setText(QString::fromStdString(std::to_string(sp.max_reproducing_distance)));
    _ui.le_min_move_range                    ->setText(QString::fromStdString(std::to_string(sp.min_move_range)));
    _ui.le_max_move_range                    ->setText(QString::fromStdString(std::to_string(sp.max_move_range)));

    _ui.cb_reproducing_rotation_enabled      ->setChecked(sp.reproduction_rotation_enabled);
    _ui.cb_runtime_rotation_enabled          ->setChecked(sp.runtime_rotation_enabled);
    _ui.cb_on_touch_kill                     ->setChecked(sp.on_touch_kill);
    _ui.cb_movers_can_produce_food           ->setChecked(sp.movers_can_produce_food);
    _ui.cb_food_blocks_reproduction          ->setChecked(sp.food_blocks_reproduction);
    _ui.cb_food_blocks_movement              ->setChecked(sp.food_blocks_movement);
    _ui.cb_fix_reproduction_distance         ->setChecked(sp.reproduction_distance_fixed);
    _ui.cb_use_evolved_brain_mutation_rate   ->setChecked(sp.use_brain_evolved_mutation_rate);
    _ui.cb_use_evolved_anatomy_mutation_rate ->setChecked(sp.use_anatomy_evolved_mutation_rate);
    _ui.cb_disable_warnings                  ->setChecked(disable_warnings);
    _ui.cb_self_organism_blocks_block_sight  ->setChecked(sp.organism_self_blocks_block_sight);
    _ui.cb_set_fixed_move_range              ->setChecked(sp.set_fixed_move_range);
    _ui.cb_failed_reproduction_eats_food     ->setChecked(sp.failed_reproduction_eats_food);
    _ui.cb_rotate_every_move_tick            ->setChecked(sp.rotate_every_move_tick);
    _ui.cb_multiply_food_production_prob     ->setChecked(sp.multiply_food_production_prob);
    _ui.cb_simplified_food_production        ->setChecked(sp.simplified_food_production);
    _ui.cb_stop_when_one_food_generated      ->setChecked(sp.stop_when_one_food_generated);
    _ui.cb_eat_then_produce                  ->setChecked(sp.eat_then_produce);
    _ui.cb_use_new_child_pos_calculator      ->setChecked(sp.use_new_child_pos_calculator);
    _ui.cb_checks_if_path_is_clear           ->setChecked(sp.check_if_path_is_clear);

    //Settings
    _ui.le_perlin_persistence->setText(QString::fromStdString(to_str(sp.perlin_persistence, 3)));
    _ui.le_perlin_upper_bound->setText(QString::fromStdString(to_str(sp.perlin_upper_bound, 3)));
    _ui.le_perlin_lower_bound->setText(QString::fromStdString(to_str(sp.perlin_lower_bound, 3)));
    _ui.le_perlin_x_modifier ->setText(QString::fromStdString(to_str(sp.perlin_x_modifier,  3)));
    _ui.le_perlin_y_modifier ->setText(QString::fromStdString(to_str(sp.perlin_y_modifier,  3)));
    _ui.le_keyboard_movement_amount->setText(QString::fromStdString(to_str(keyboard_movement_amount, 1)));
    _ui.le_scaling_coefficient->setText(QString::fromStdString(to_str(scaling_coefficient, 1)));

    _ui.le_num_threads->setText(QString::fromStdString(std::to_string(ecp.num_threads)));
    _ui.le_float_number_precision->setText(QString::fromStdString(std::to_string(float_precision)));
    _ui.le_perlin_octaves->setText(QString::fromStdString(std::to_string(sp.perlin_octaves)));
    //font size could be set either by pixel_size or point_size. If it is set by one, the other will give -1
    int font_size;
    if (font().pixelSize() < 0) {
        font_size = font().pointSize();
    } else {
        font_size = font().pixelSize();
    }
    _ui.le_font_size              ->setText(QString::fromStdString(std::to_string(font_size)));

//    _ui.rb_partial_multi_thread_mode->hide();
    _ui.rb_multi_thread_mode->hide();
    _ui.rb_cuda_mode->hide();

    _ui.table_organism_block_parameters->horizontalHeader()->setVisible(true);
    _ui.table_organism_block_parameters->verticalHeader()->setVisible(true);
    _ui.cb_wait_for_engine_to_stop->setChecked(wait_for_engine_to_stop_to_render);
    _ui.cb_simplified_rendering->setChecked(simplified_rendering);

    _ui.le_update_info_every_n_milliseconds ->setText(QString::fromStdString(std::to_string(update_info_every_n_milliseconds)));
    _ui.cb_synchronise_info_with_window->setChecked(synchronise_info_update_with_window_update);
    disable_warnings = true;
    _ui.cb_use_nvidia_for_image_generation->setChecked(use_cuda);
    disable_warnings = false;
    _ui.le_menu_height->setText(QString::fromStdString(std::to_string(_ui.menu_frame->frameSize().height())));
    _ui.cb_really_stop_render->setChecked(really_stop_render);
#if __CUDA_USED__ == 0
    _ui.cb_use_nvidia_for_image_generation->hide();
#endif

    //So that when user clicks on rbs in organism editors, rbs in main window would be unchecked and vice versa
    _ui.rb_null_button->hide();
    ee._ui.rb_null_button->hide();

    //TODO
    _ui.rb_single_thread_mode->hide();
    _ui.rb_partial_multi_thread_mode->hide();
    _ui.le_num_threads->hide();
//    _ui.tb_open_organism_editor->setEnabled(false);
//    _ui.cb_editor_always_on_top->setEnabled(false);
    _ui.lb_set_num_threads->hide();
}

void MainWindow::update_simulation_size_label() {
  s._ui.lb_simulation_size->setText(QString::fromStdString("Simulation size: " + std::to_string(edc.simulation_width) + "x" + std::to_string(edc.simulation_height)));
}

std::string MainWindow::convert_num_bytes(uint64_t num_bytes) {
    uint64_t previous = num_bytes;
    num_bytes /= 1024;
    if (!num_bytes) {return std::to_string(previous) + " B";}
    previous = num_bytes;
    num_bytes /= 1024;
    if (!num_bytes) {return std::to_string(previous) + "KiB";}
    previous = num_bytes;
    num_bytes /= 1024;
    if (!num_bytes) {return std::to_string(previous) + "MiB";}
    previous = num_bytes;
    num_bytes /= 1024;
    if (!num_bytes) {return std::to_string(previous) + "GiB";}

    return std::to_string(num_bytes) + " TiB";
}

Vector2<int> MainWindow::calculate_cursor_pos_on_grid(int x, int y) {
    auto c_pos = Vector2<int>{};
    c_pos.x = static_cast<int>((x - float(_ui.simulation_graphicsView->viewport()->width() )/2)*scaling_zoom + center_x);
    c_pos.y = static_cast<int>((y - float(_ui.simulation_graphicsView->viewport()->height())/2)*scaling_zoom + center_y);
    return c_pos;
}

void MainWindow::change_main_grid_left_click() {
    //cursor Vector2 on grid
    auto cpg = calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    ecp.pause_processing_user_action = true;
    wait_for_engine_to_pause_processing_user_actions();
    for (int x = -brush_size / 2; x < float(brush_size) / 2; x++) {
        for (int y = -brush_size / 2; y < float(brush_size) / 2; y++) {
            switch (cursor_mode) {
                case CursorMode::ModifyFood:
                    edc.user_actions_pool.emplace_back(ActionType::TryAddFood, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::ModifyWall:
                    edc.user_actions_pool.emplace_back(ActionType::TryAddWall, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::KillOrganism:
                    edc.user_actions_pool.emplace_back(ActionType::TryKillOrganism, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::ChooseOrganism:
                    edc.user_actions_pool.emplace_back(ActionType::TrySelectOrganism, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::PlaceOrganism:
                    edc.user_actions_pool.emplace_back(ActionType::TryAddOrganism, cpg.x, cpg.y);
                    goto endfor;
            }
        }
    }
    endfor:

    ecp.pause_processing_user_action = false;
}

void MainWindow::change_main_grid_right_click() {
    auto cpg = calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    ecp.pause_processing_user_action = true;
    wait_for_engine_to_pause_processing_user_actions();
    for (int x = -brush_size/2; x < float(brush_size)/2; x++) {
        for (int y = -brush_size/2; y < float(brush_size)/2; y++) {
            switch (cursor_mode) {
                case CursorMode::ModifyFood:
                    edc.user_actions_pool.emplace_back(ActionType::TryRemoveFood, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::ModifyWall:
                    edc.user_actions_pool.emplace_back(ActionType::TryRemoveWall, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::KillOrganism:
                    edc.user_actions_pool.emplace_back(ActionType::TryKillOrganism, cpg.x + x, cpg.y + y);
                    break;
                case CursorMode::ChooseOrganism:
                    break;
                case CursorMode::PlaceOrganism:
                    break;
            }
        }
    }
    ecp.pause_processing_user_action = false;
}

void MainWindow::change_editing_grid_left_click() {
    auto cpg = ee.calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    if (cpg.x < 0 || cpg.y < 0 || cpg.x >= ee.editor_width || cpg.y >= ee.editor_height) { return;}

    //relative position
    auto r_pos = Vector2<int>{cpg.x - ee.editor_organism->x, cpg.y - ee.editor_organism->y};
    ee.editor_organism->anatomy->set_block(ee.chosen_block_type, ee.chosen_block_rotation, r_pos.x, r_pos.y);
    ee.create_image();
}

void MainWindow::change_editing_grid_right_click() {
    auto cpg = ee.calculate_cursor_pos_on_grid(last_mouse_x_pos, last_mouse_y_pos);
    if (cpg.x < 0 || cpg.y < 0 || cpg.x >= ee.editor_width || cpg.y >= ee.editor_height) { return;}
    if (cpg.x == ee.editor_organism->x && cpg.y == ee.editor_organism->y) {return;}
    if (ee.editor_organism->anatomy->_organism_blocks.size() == 1) { return;}

    //relative position
    auto r_pos = Vector2<int>{cpg.x - ee.editor_organism->x, cpg.y - ee.editor_organism->y};
    ee.editor_organism->anatomy->set_block(BlockTypes::EmptyBlock, Rotation::UP, r_pos.x, r_pos.y);
    ee.create_image();
}

bool MainWindow::cuda_is_available() {
#if __CUDA_USED__
    auto count = get_device_count();
    if (count <= 0) {
        return false;
    } else {
        return true;
    }
#else
    return false;
#endif
}