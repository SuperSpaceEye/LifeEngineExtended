//
// Created by spaceeye on 16.03.2022.
//

#include "WindowCore.h"

WindowCore::WindowCore(int window_width, int window_height, int simulation_width, int simulation_height, int window_fps,
                       int simulation_fps, int simulation_num_threads, QWidget *parent) :
        QWidget(parent), window_width(window_width), window_height(window_height), max_window_fps(window_fps), max_simulation_fps(simulation_fps) {
    _ui.setupUi(this);

    _ui.graphicsView->show();

    //https://stackoverflow.com/questions/32714105/mousemoveevent-is-not-called
    QCoreApplication::instance()->installEventFilter(this);

    set_window_interval();
    set_simulation_interval();

    dc.simulation_width = simulation_width;
    dc.simulation_height = simulation_height;

    set_simulation_num_threads(simulation_num_threads);

    dc.simulation_grid.resize(simulation_width, std::vector<BaseGridBlock>(simulation_height, BaseGridBlock{}));
    dc.second_simulation_grid.resize(simulation_width, std::vector<BaseGridBlock>(simulation_height, BaseGridBlock{}));
    //engine = new SimulationEngine(simulation_width, simulation_height, std::ref(simulation_grid), std::ref(engine_ticks), std::ref(engine_working), std::ref(engine_paused), std::ref(engine_mutex));
    engine = new SimulationEngine(std::ref(dc), std::ref(cp), std::ref(op), engine_mutex);

//    std::random_device rd;
//    std::mt19937 mt(rd());
//    std::uniform_int_distribution<int> dist(0, 8);
//
//    for (int relative_x = 0; relative_x < simulation_width; relative_x++) {
//        for (int relative_y = 0; relative_y < simulation_height; relative_y++) {
//            simulation_grid[relative_x][relative_y].type = static_cast<BlockTypes>(dist(mt));
//        }
//    }

    color_container = ColorContainer{};

    resize_image();
    reset_image();


    QTimer::singleShot(0, [&]{
        engine_thread = std::thread{&SimulationEngine::threaded_mainloop, engine};
        engine_thread.detach();

        start = clock.now();
        end = clock.now();

        fps_timer = clock.now();

        scene.addItem(&pixmap_item);
        _ui.graphicsView->setScene(&scene);
        std::cout << "here\n";
    });

    timer = new QTimer(parent);
    connect(timer, &QTimer::timeout, [&]{mainloop_tick();});
    timer->start();
}

void WindowCore::mainloop_tick() {
    std::this_thread::sleep_for(std::chrono::microseconds(int(window_interval * 1000000 - delta_window_processing_time)));
    start = clock.now();
    window_tick();
    window_frames++;
    end = clock.now();
    delta_window_processing_time = std::abs(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    // timer
    if (std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - fps_timer).count() / 1000. > 1) {
        //TODO make pausing logic better.
        //pauses engine, parses/loads data from/to engine, resumes engine. It looks stupid, but works
        cp.engine_pause = true;
        wait_for_engine_to_pause();
        simulation_frames = dc.engine_ticks;
        dc.engine_ticks = 0;
        cp.engine_pause = false;

        update_fps_labels(window_frames, simulation_frames);

        std::cout << window_frames << " frames in a second | " << simulation_frames << " simulation ticks in second\n";
        window_frames = 0;
        fps_timer = clock.now();
    }
}

void WindowCore::update_fps_labels(int fps, int sps) {
    _ui.fps_label->setText(QString::fromStdString("fps: " + std::to_string(fps)));
    _ui.sps_label->setText(QString::fromStdString("sps: "+std::to_string(sps)));
}

void WindowCore::window_tick() {
//    //TODO this needs refactoring
//    sf::Event event;
//    while (window.pollEvent(event)) {
//        gui.handleEvent(event);
//
//        if (event.type == sf::Event::Closed) {
//            cp.stop_engine = true;
//            window.close();
//        } else if (event.type == sf::Event::MouseWheelScrolled) {
//            if (event.mouseWheelScroll.delta == 1) {
//                scaling_zoom /= scaling_coefficient;
//            } else {
//                scaling_zoom *= scaling_coefficient;
//            }
//        } else if (event.type == sf::Event::MouseButtonPressed) {
//            //TODO make middle click
//            if (event.mouseButton.button == sf::Mouse::Right) {
//                auto position = sf::Mouse::getPosition(window);
//
//                if (canvas->isMouseOnWidget(tgui::Vector2f{(float)position.x, (float)position.y})) {
//                    right_mouse_button_pressed = true;
//                }
//                last_mouse_x = position.x;
//                last_mouse_y = position.y;
//            } else if (event.mouseButton.button == sf::Mouse::Left) {
//                left_mouse_button_pressed = true;
//            }
//        } else if (event.type == sf::Event::MouseButtonReleased) {
//            if (event.mouseButton.button == sf::Mouse::Right) {
//                right_mouse_button_pressed = false;
//            } else if (event.mouseButton.button == sf::Mouse::Left) {
//                left_mouse_button_pressed = false;
//            }
//        } else if (event.type == sf::Event::MouseMoved) {
//            if (right_mouse_button_pressed) {
//                int delta_x = event.mouseMove.x - last_mouse_x;
//                int delta_y = event.mouseMove.y - last_mouse_y;
//
//                move_center(delta_x, delta_y);
//
//                last_mouse_x = event.mouseMove.x;
//                last_mouse_y = event.mouseMove.y;
//            }
//        } else if (event.type == sf::Event::KeyPressed) {
//            if (event.key.code == sf::Keyboard::R) {
//                reset_image();
//            } else if (event.key.code == sf::Keyboard::M) {
//                menu_shows = !menu_shows;
//                configure_simulation_canvas();
//                configure_menu_canvas();
//            } else if (event.key.code == sf::Keyboard::Space) {
//                cp.engine_global_pause = !cp.engine_global_pause;
//                parse_full_simulation_grid(cp.engine_global_pause);
//            // the ">" key
//            } else if (event.key.code == sf::Keyboard::Period) {
//                cp.engine_pass_tick = true;
//                parse_full_simulation_grid(true);
//            } else if (event.key.code == sf::Keyboard::S) {
//                pause_image_construction = !pause_image_construction;
//                parse_full_simulation_grid(pause_image_construction);
//                // calculating delta time is not needed when no image is being created.
//                cp.calculate_simulation_tick_delta_time = !cp.calculate_simulation_tick_delta_time;
//            }
//        } else if (event.type == sf::Event::Resized) {
//            configure_simulation_canvas();
//            configure_menu_canvas();
//            for (auto & layout: layouts) {
//                layout->update_positions();
//            }
//        }
//    }

//    window.clear(sf::Color(0, 0, 0, 255));
//    make_image();
//    gui.draw();
//    window.display();
    make_image();
}

void WindowCore::resize_image() {
    image_vector.clear();
    image_vector.resize(4 * _ui.graphicsView->viewport()->width() * _ui.graphicsView->viewport()->height(), 255);
}

void WindowCore::move_center(int delta_x, int delta_y) {
    center_x -= delta_x * scaling_zoom;
    center_y -= delta_y * scaling_zoom;
}

void WindowCore::reset_image() {
    center_x = (float)dc.simulation_width/2;
    center_y = (float)dc.simulation_height/2;
    // finds exponent needed to scale the sprite
    auto exp = log((float)start_height/window_height) / log(1.05);
    scaling_zoom = pow(scaling_coefficient, exp);
}

// TODO it is probably possible to do better.
QColor& WindowCore::get_color(BlockTypes type) {
    switch (type) {
        case EmptyBlock :   return color_container.empty_block;
        case MouthBlock:    return color_container.mouth;
        case ProducerBlock: return color_container.producer;
        case MoverBlock:    return color_container.mover;
        case KillerBlock:   return color_container.killer;
        case ArmorBlock:    return color_container.armor;
        case EyeBlock:      return color_container.eye;
        case FoodBlock:     return color_container.food;
        case WallBlock:     return color_container.wall;
        default: return color_container.empty_block;
    }
}

//TODO bug. it has double pixel row and column at the left and up boundaries.
void WindowCore::create_image() {
    resize_image();
    auto image_width = _ui.graphicsView->viewport()->width();
    auto image_height = _ui.graphicsView->viewport()->height();

    image = QImage(image_width, image_height, QImage::Format_RGB32);

    int scaled_width = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and stop coordinates on simulation grid
    auto start_x = (int)(center_x-(scaled_width / 2));
    auto end_x = (int)(center_x+(scaled_width / 2));

    auto start_y = (int)(center_y-(scaled_height / 2));
    auto end_y = (int)(center_y+(scaled_height / 2));

    //TODO possible optimization by caching result from linspace
    auto lin_width = Linspace<int>()(start_x, end_x, image_width);
    auto lin_height = Linspace<int>()(start_y, end_y, image_height);

    std::vector<int> truncated_lin_width; truncated_lin_width.reserve(image_width);
    std::vector<int> truncated_lin_height; truncated_lin_height.reserve(image_height);

    int min_val = -1;
    for (int x = 0; x < image_width; x++) {
        if (lin_width[x] > min_val) {min_val = lin_width[x]; truncated_lin_width.push_back(min_val);}}
    min_val = -1;
    for (int y = 0; y < image_height; y++) {if (lin_height[y] > min_val) {min_val = lin_height[y]; truncated_lin_height.push_back(min_val);}}

    if (!pause_image_construction && !cp.engine_global_pause) {
        //it does not help
        //#pragma omp parallel for
        cp.engine_pause = true;
        // pausing engine to parse data from engine.
        auto paused = wait_for_engine_to_pause();
        // if for some reason engine is not paused in time, it will use old parsed data and not switch engine on.
        if (paused) {parse_simulation_grid(truncated_lin_width, truncated_lin_height); cp.engine_pause = false;}
    }

    auto pixel_color = color_container.empty_block;
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            //TODO refactor this
            if      (lin_width[x] < 0 || lin_width[x] >= dc.simulation_width)    {pixel_color = color_container.simulation_background_color;}
            else if (lin_height[y] < 0 || lin_height[y] >= dc.simulation_height) {pixel_color = color_container.simulation_background_color;}
            else                                                                 {pixel_color = get_color(dc.second_simulation_grid[lin_width[x]][lin_height[y]].type);}
            if (!compare_pixel_color(x, y, pixel_color)) {set_image_pixel(x, y, pixel_color);}
        }
    }
    pixmap_item.setPixmap(QPixmap::fromImage(QImage(image_vector.data(), image_width, image_height, QImage::Format_RGB32)));
}

void WindowCore::set_image_pixel(int x, int y, QColor &color) {
    image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)  ] = color.red();
    image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)+1] = color.green();
    image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)+2] = color.blue();
}

bool WindowCore::compare_pixel_color(int x, int y, QColor &color) {
    return image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)  ] == color.red() *
           image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)+1] == color.green() *
           image_vector[4 * (y * _ui.graphicsView->viewport()->width() + x)+2] == color.blue();
}

void WindowCore::make_image() {
    create_image();
}

void WindowCore::set_window_interval() {
    if (max_window_fps <= 0) {
        window_interval = 0.;
        return;
    }
    window_interval = 1./max_window_fps;
}

void WindowCore::set_simulation_interval() {
    if (max_simulation_fps <= 0) {
        dc.simulation_interval = 0.;
        dc.unlimited_simulation_fps = true;
        return;
    }
    dc.simulation_interval = 1./max_simulation_fps;
    dc.unlimited_simulation_fps = false;
}

// __attribute__((optimize("O0")))
bool WindowCore::wait_for_engine_to_pause() {
    // if engine is paused by user, then return if engine is really paused.
    if (cp.engine_global_pause) {return cp.engine_paused;}
    auto sleeping_time = std::chrono::microseconds (int(dc.delta_time*1.5));
    // if sleeping time is larger than spare window processing time, then not wait and return result straight away.
    //TODO not needed
    if (!allow_waiting_overhead) {
        if (sleeping_time.count() > int(window_interval * std::chrono::microseconds::period::den) - delta_window_processing_time) { return cp.engine_paused; }
    } else {
        if (sleeping_time.count() > int(window_interval * std::chrono::microseconds::period::den)) { return cp.engine_paused; }
    }
    std::this_thread::sleep_for(sleeping_time);
    return cp.engine_paused;
}

void WindowCore::parse_simulation_grid(std::vector<int> lin_width, std::vector<int> lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= dc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= dc.simulation_height) { continue; }
            dc.second_simulation_grid[x][y].type = dc.simulation_grid[x][y].type;
        }
    }
}

void WindowCore::parse_full_simulation_grid(bool parse) {
    if (!parse) {return;}
    full_simulation_grid_parsed = true;

    cp.engine_pause = true;
    while(!wait_for_engine_to_pause()) {}

    for (int x = 0; x < dc.simulation_width; x++) {
        for (int y = 0; y < dc.simulation_height; y++) {
            dc.second_simulation_grid[x][y].type = dc.simulation_grid[x][y].type;
        }
    }
    cp.engine_pause = false;
}

void WindowCore::set_simulation_num_threads(uint8_t num_threads) {
    cp.num_threads = num_threads;
    cp.build_threads = true;
}

//====================Events====================

void WindowCore::closeEvent(QCloseEvent *event) {
    cp.stop_engine = true;
    QWidget::closeEvent(event);
}

void WindowCore::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::RightButton) {
        auto position = event->pos();

        if (_ui.graphicsView->underMouse()) {
            right_mouse_button_pressed = true;
        }

        last_mouse_x = position.x();
        last_mouse_y = position.y();
    } else if (event->button() == Qt::LeftButton) {
        left_mouse_button_pressed = true;
    }
}


bool WindowCore::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::MouseMove) {
        auto mouse_event = dynamic_cast<QMouseEvent*>(event);
        if (right_mouse_button_pressed) {
            int delta_x = mouse_event->x() - last_mouse_x;
            int delta_y = mouse_event->y() - last_mouse_y;

            move_center(delta_x, delta_y);

            last_mouse_x = mouse_event->x();
            last_mouse_y = mouse_event->y();
        }
    } else if (event->type() == QEvent::MouseButtonRelease) {
        auto mouse_event = dynamic_cast<QMouseEvent*>(event);
        if (mouse_event->button() == Qt::RightButton) {
            right_mouse_button_pressed = false;
        } else if (mouse_event->button() == Qt::LeftButton) {
            left_mouse_button_pressed = false;
        }
    }
    return false;
}

void WindowCore::wheelEvent(QWheelEvent *event) {
    if (event->delta() > 0) {
        scaling_zoom /= scaling_coefficient;
    } else {
        scaling_zoom *= scaling_coefficient;
    }
}

//====================SLOTS====================

void WindowCore::pause_slot(bool paused) {
    cp.engine_global_pause = !cp.engine_global_pause;
    parse_full_simulation_grid(cp.engine_global_pause);
}

void WindowCore::stoprender_slot(bool stopped_render) {
    pause_image_construction = !pause_image_construction;
    parse_full_simulation_grid(pause_image_construction);
    // calculating delta time is not needed when no image is being created.
    cp.calculate_simulation_tick_delta_time = !cp.calculate_simulation_tick_delta_time;
}

void WindowCore::clear_slot() {

}

void WindowCore::reset_slot() {

}

void WindowCore::pass_one_tick_slot() {
    cp.engine_pass_tick = true;
    parse_full_simulation_grid(true);
}
void WindowCore::reset_view_slot() {
    reset_image();
}