//
// Created by spaceeye on 16.03.2022.
//

//TODO There is always 14 memory leaks. Probably driver's problem.

#include "UIWindow.h"

UIWindow::UIWindow(int window_width, int window_height, int simulation_width, int simulation_height, int window_fps,
                   int simulation_fps) : window_width(window_width), window_height(window_height),
                                         max_window_fps(window_fps), max_simulation_fps(simulation_fps) {
    settings = sf::ContextSettings();
    settings.antialiasingLevel = 0;

    window.create(sf::VideoMode(window_width, window_height), "TheLifeEngineCpp", sf::Style::Default, settings);
    window.setFramerateLimit(0);
    gui.setTarget(window);

    set_window_interval();
    set_simulation_interval();

    dc.simulation_width = simulation_width;
    dc.simulation_height = simulation_height;

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

    layouts = std::vector<GridLayout2D*>();

    color_container = ColorContainer{};

    make_visuals();

    reset_image();
}

void UIWindow::main_loop() {
    if (separate_process) {
        multi_thread_main_loop();
    } else {
        single_thread_main_loop();
    }
}

void UIWindow::multi_thread_main_loop() {
    engine_thread = std::thread{&SimulationEngine::threaded_mainloop, engine};
    engine_thread.detach();

    //TODO limits fps by sleep(window_interval-time_to_process_one_frame).
    auto start = clock.now();
    auto end = clock.now();

    fps_timer = clock.now();
    while (window.isOpen()) {
        std::this_thread::sleep_for(std::chrono::microseconds(int(window_interval*1000000 - delta_window_processing_time)));
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
}

void UIWindow::single_thread_main_loop() {
//    if (!unlimited_window_fps) {last_window_update = clock.now();}
//    if (!dc.unlimited_simulation_fps) {last_simulation_update = clock.now();}
//    fps_timer = clock.now();
//    while (window.isOpen()) {
//        if (unlimited_window_fps || std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - last_window_update).count() / 1000000. >= window_interval) {
//            window_tick();
//            window_frames++;
//            last_window_update = clock.now();
//        }
//        if (dc.unlimited_simulation_fps || std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - last_simulation_update).count() / 1000000. >= simulation_interval) {
//            engine->simulation_tick();
//            simulation_frames++;
//            last_simulation_update = clock.now();
//        }
//        // timer
//        if (std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - fps_timer).count() / 1000. > 1) {
//            update_fps_labels(window_frames, simulation_frames);
//
//            std::cout << window_frames << " frames in a second | " << simulation_frames << " simulation ticks in second\n";
//            window_frames = 0;
//            simulation_frames = 0;
//            fps_timer = clock.now();
//        }
//    }
}

void UIWindow::update_fps_labels(int fps, int sps) {
    window_fps_label->setText("fps: "+std::to_string(fps));
    simulation_fps_label->setText("sps: "+std::to_string(sps));
}

void UIWindow::window_tick() {
    //TODO this needs refactoring
    sf::Event event;
    while (window.pollEvent(event)) {
        gui.handleEvent(event);

        if (event.type == sf::Event::Closed) {
            cp.engine_working = false;
            window.close();
            //TODO detached thread doesn't want to stop. oh well.
            exit(0);
        } else if (event.type == sf::Event::MouseWheelScrolled) {
            if (event.mouseWheelScroll.delta == 1) {
                scaling_zoom /= scaling_coefficient;
            } else {
                scaling_zoom *= scaling_coefficient;
            }
        } else if (event.type == sf::Event::MouseButtonPressed) {
            //TODO make middle click
            if (event.mouseButton.button == sf::Mouse::Right) {
                auto position = sf::Mouse::getPosition(window);

                if (canvas->isMouseOnWidget(tgui::Vector2f{(float)position.x, (float)position.y})) {
                    right_mouse_button_pressed = true;
                }
                last_mouse_x = position.x;
                last_mouse_y = position.y;
            } else if (event.mouseButton.button == sf::Mouse::Left) {
                left_mouse_button_pressed = true;
            }
        } else if (event.type == sf::Event::MouseButtonReleased) {
            if (event.mouseButton.button == sf::Mouse::Right) {
                right_mouse_button_pressed = false;
            } else if (event.mouseButton.button == sf::Mouse::Left) {
                left_mouse_button_pressed = false;
            }
        } else if (event.type == sf::Event::MouseMoved) {
            if (right_mouse_button_pressed) {
                int delta_x = event.mouseMove.x - last_mouse_x;
                int delta_y = event.mouseMove.y - last_mouse_y;

                move_center(delta_x, delta_y);

                last_mouse_x = event.mouseMove.x;
                last_mouse_y = event.mouseMove.y;
            }
        } else if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::R) {
                reset_image();
            } else if (event.key.code == sf::Keyboard::M) {
                menu_shows = !menu_shows;
                configure_simulation_canvas();
                configure_menu_canvas();
            } else if (event.key.code == sf::Keyboard::Space) {
                cp.engine_global_pause = !cp.engine_global_pause;
            // the ">" key
            } else if (event.key.code == sf::Keyboard::Period) {
                cp.engine_pass_tick = true;
            } else if (event.key.code == sf::Keyboard::S) {
                pause_image_construction = !pause_image_construction;
                parse_full_simulation_grid(pause_image_construction);
                // calculating delta time is not needed when no image is being created.
                cp.calculate_simulation_tick_delta_time = !cp.calculate_simulation_tick_delta_time;
            }
        } else if (event.type == sf::Event::Resized) {
            configure_simulation_canvas();
            configure_menu_canvas();
            for (auto & layout: layouts) {
                layout->update_positions();
            }
        }
    }
    window.clear(sf::Color(0, 0, 0, 255));
    //todo delete in the future?
    make_image();
    gui.draw();
    window.display();
}

void UIWindow::configure_simulation_canvas() {
    image_width = window.getSize().x;
    image_height = window.getSize().y;
    if (menu_shows){
        canvas->setSize(window.getSize().x, window.getSize().y * (1 - menu_size));
    } else {
        canvas->setSize(window.getSize().x, window.getSize().y);
    }
}

void UIWindow::configure_menu_canvas() {
    if (!menu_shows) {
        menu_canvas->setEnabled(false);
        menu_canvas->setVisible(false);
        for (auto & layout : layouts) {
            layout->setEnabled(false);
            layout->setVisible(false);
        }
        for (auto & widget: menu_widgets) {
            widget->setEnabled(false);
            widget->setVisible(false);
        }
        return;
    } else {
        menu_canvas->setEnabled(true);
        menu_canvas->setVisible(true);
        for (auto & layout : layouts) {
            layout->setEnabled(true);
            layout->setVisible(true);
        }
        for (auto & widget: menu_widgets) {
            widget->setEnabled(true);
            widget->setVisible(true);
        }
    }
    menu_sprite.setScale(window.getSize().x, window.getSize().y*menu_size);
    menu_canvas->setSize(window.getSize().x, window.getSize().y * menu_size);
    menu_canvas->setPosition(0, (1 - menu_size) * window.getSize().y);
    menu_canvas->draw(menu_sprite);
    menu_canvas->display();
}

// i hate making gui's
void UIWindow::make_visuals() {
    canvas = tgui::CanvasSFML::create({window_width, window_height});
    configure_simulation_canvas();

    menu_canvas = tgui::CanvasSFML::create({window_width, window_height});
    menu_texture = sf::Texture{};
    menu_image = sf::Image{};
    menu_image.create(1, 1);
    menu_image.setPixel(0, 0, color_container.menu_color);
    menu_texture.loadFromImage(menu_image);
    menu_sprite = sf::Sprite{};
    menu_sprite.setTexture(menu_texture);
    menu_sprite.setScale(1, 1);
    //why do i need this? idk.
    menu_canvas->setSize({"100%", "100%"});
    configure_menu_canvas();

    gui.add(canvas);
    gui.add(menu_canvas);

    menu_canvas->draw(menu_sprite);
    menu_canvas->display();

    auto layout = tgui::VerticalLayout::create();

    layout->setPosition({tgui::bindPosX(menu_canvas) + 10, tgui::bindPosY(menu_canvas) + 10});
    layout->setSize("100%", "10%");

    window_fps_label = tgui::Label::create();
    simulation_fps_label = tgui::Label::create();

    window_fps_label->setText("None");
    window_fps_label->setTextSize(20);
    simulation_fps_label->setText("None");
    simulation_fps_label->setTextSize(20);

    layout->add(window_fps_label);
    layout->add(simulation_fps_label);

    gui.add(layout);
    menu_widgets.push_back(layout);

    auto tabs = tgui::Tabs::create();
    tabs->setTabHeight(30);
    tabs->setPosition({tgui::bindPosX(layout)+150, tgui::bindPosY(layout)+10});
    tabs->add("Tab 1");
    tabs->add("Tab 2");
    tabs->add("Tab 3");
    gui.add(tabs);
    menu_widgets.push_back(tabs);


//    auto panel = tgui::Panel::create();
//    panel->setPosition(600, 300);
//    panel->setSize(300,300);
////    auto rend = tgui::PanelRenderer{};
////    rend.setTextureBackground()
////
////    panel->setRenderer(rend.getData());
//    gui.add(panel);

//    auto button = tgui::Button::create();
//    button->setText("Reset simulation");
//    button->onPress([&]{});
//    button->setPosition({tgui::bindPosX(layout), tgui::bindPosY(layout)+80});
//    gui.add(button);
//    menu_widgets.push_back(button);
//
//    auto button2 = tgui::Button::create();
//    button2->setText("Reset view");
//    button2->onPress([&]{reset_image();});
//    button2->setPosition({tgui::bindPosX(button), tgui::bindPosY(button)+30});
//    gui.add(button2);
//    menu_widgets.push_back(button2);

}

void UIWindow::move_center(int delta_x, int delta_y) {
    center_x -= delta_x * scaling_zoom;
    center_y -= delta_y * scaling_zoom;
}

void UIWindow::reset_image() {
    center_x = (float)dc.simulation_width/2;
    center_y = (float)dc.simulation_height/2;
    // finds exponent needed to scale the sprite
    auto exp = log((float)start_height/window_height) / log(1.05);
    scaling_zoom = pow(scaling_coefficient, exp);
}

// TODO it is probably possible to do better.
sf::Color& UIWindow::get_color(BlockTypes type) {
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
void UIWindow::create_image() {
    // very important
    sprite = sf::Sprite{};
    image.create(image_width, image_height, color_container.simulation_background_color);

    int scaled_width = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and stop coordinates on simulation grid
    auto start_x = (int)(center_x-(scaled_width / 2));
    auto end_x = (int)(center_x+(scaled_width / 2));

    auto start_y = (int)(center_y-(scaled_height / 2));
    auto end_y = (int)(center_y+(scaled_height / 2));

    //TODO possible optimization by caching result from linspace
    auto lin_width = linspace(start_x, end_x, image_width);
    auto lin_height = linspace(start_y, end_y, image_height);

    std::vector<int> truncated_lin_width; truncated_lin_width.reserve(image_width);
    std::vector<int> truncated_lin_height; truncated_lin_height.reserve(image_height);

    int min_val = -1;
    for (int x = 0; x < image_width; x++) {
        if (int(lin_width[x]) > min_val) {min_val = int(lin_width[x]); truncated_lin_width.push_back(min_val);}}
    min_val = -1;
    for (int y = 0; y < image_height; y++) {if (int(lin_height[y]) > min_val) {min_val = int(lin_height[y]); truncated_lin_height.push_back(min_val);}}

    if (!pause_image_construction && !cp.engine_global_pause) {
        //it does not help
        //#pragma omp parallel for
        cp.engine_pause = true;
        // pausing engine to parse data from engine.
        auto paused = wait_for_engine_to_pause();
        // if for some reason engine is not paused in time, it will use old parsed data and not switch engine on.
        if (paused) {parse_simulation_grid(truncated_lin_width, truncated_lin_height); cp.engine_pause = false;}
    }

    for (int x = 0; x < image_width; x++) {
        if ((int) lin_width[x] < 0 || (int) lin_width[x] >= dc.simulation_width) { continue; }
        for (int y = 0; y < image_height; y++) {
            if ((int) lin_height[y] < 0 || (int) lin_height[y] >= dc.simulation_height) { continue; }
            image.setPixel(x, y, get_color(dc.second_simulation_grid[(int) lin_width[x]][(int) lin_height[y]].type));
        }
    }

    texture.loadFromImage(image);
    sprite.setTexture(texture);
}

void UIWindow::make_image() {
    create_image();

    canvas->clear(color_container.simulation_background_color);
    canvas->draw(sprite);
    //TODO bug. rarely throws malloc for some reason.
    // and now I can't catch it. great.

    //TODO second bug. with canvas displaying, the fps is lower for some reason. And it can run faster, it just doesn't want to.
    canvas->display();
}

void UIWindow::set_window_interval() {
    if (max_window_fps <= 0) {
        window_interval = 0.;
        unlimited_window_fps = true;
        return;
    }
    window_interval = 1./max_window_fps;
    unlimited_window_fps = false;
}

void UIWindow::set_simulation_interval() {
    if (max_simulation_fps <= 0) {
        dc.simulation_interval = 0.;
        dc.unlimited_simulation_fps = true;
        return;
    }
    dc.simulation_interval = 1./max_simulation_fps;
    dc.unlimited_simulation_fps = false;
}

// __attribute__((optimize("O0")))
bool UIWindow::wait_for_engine_to_pause() {
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

void UIWindow::parse_simulation_grid(std::vector<int> lin_width, std::vector<int> lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= dc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= dc.simulation_height) { continue; }
            dc.second_simulation_grid[x][y].type = dc.simulation_grid[x][y].type;
        }
    }
}

void UIWindow::parse_full_simulation_grid(bool parse) {
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

std::vector<double> UIWindow::linspace(double start, double end, int num) {
    std::vector<double> linspaced;
    linspaced.reserve(num);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}
