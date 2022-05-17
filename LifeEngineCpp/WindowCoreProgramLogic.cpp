//
// Created by spaceeye on 16.03.2022.
//

#include <memory>

#include "WindowCore.h"

WindowCore::WindowCore(int simulation_width, int simulation_height, int window_fps,
                       int simulation_fps, int simulation_num_threads, QWidget *parent) :
        QWidget(parent){
    _ui.setupUi(this);

    _ui.simulation_graphicsView->show();
    //If not false, the graphics view allows scrolling of an image after window resizing and only this helps.
    //Disabling graphics view doesn't change anything anyway.
    _ui.simulation_graphicsView->setEnabled(false);

    //https://stackoverflow.com/questions/32714105/mousemoveevent-is-not-called
    QCoreApplication::instance()->installEventFilter(this);

    dc.simulation_width = simulation_width;
    dc.simulation_height = simulation_height;

    set_simulation_num_threads(simulation_num_threads);

    dc.single_thread_simulation_grid.resize(simulation_width, std::vector<BaseGridBlock>(simulation_height, BaseGridBlock{}));
    dc.second_simulation_grid.resize(simulation_width, std::vector<BaseGridBlock>(simulation_height, BaseGridBlock{}));
    engine = new SimulationEngine(std::ref(dc), std::ref(cp), std::ref(op), std::ref(sp), engine_mutex);

    make_walls();

    std::random_device rd;
    mt = std::mt19937(rd());
//    std::uniform_int_distribution<int> dist(0, 8);
//
//    for (int relative_x = 0; relative_x < simulation_width; relative_x++) {
//        for (int relative_y = 0; relative_y < simulation_height; relative_y++) {
//            single_thread_simulation_grid[relative_x][relative_y].type = static_cast<BlockTypes>(dist(mt));
//        }
//    }

    color_container = ColorContainer{};
    sp = SimulationParameters{};

    auto anatomy = std::make_shared<Anatomy>();
    anatomy->set_block(BlockTypes::MouthBlock, 0, 0);
    anatomy->set_block(BlockTypes::ProducerBlock, -1, -1);
    anatomy->set_block(BlockTypes::ProducerBlock, 1, 1);

    auto brain = std::make_shared<Brain>(&mt, BrainTypes::RandomActions);

    //TODO very important. organism calls destructor for some reason, deallocating anatomy.
    base_organism = new Organism(dc.simulation_width / 2, dc.simulation_height / 2, &sp.reproduction_rotation_enabled,
                                 Rotation::UP, anatomy, brain, &sp, &op, &mt);
    chosen_organism = new Organism(dc.simulation_width / 2, dc.simulation_height / 2, &sp.reproduction_rotation_enabled,
                                   Rotation::UP, std::make_shared<Anatomy>(anatomy), std::make_shared<Brain>(brain),
                                   &sp, &op, &mt);

    dc.to_place_organisms.push_back(new Organism(chosen_organism));
    

    resize_image();
    reset_scale_view();

    QTimer::singleShot(0, [&]{
        engine_thread = std::thread{&SimulationEngine::threaded_mainloop, engine};
        engine_thread.detach();

        start = clock_now();
        end = clock_now();

        fps_timer = clock_now();

        scene.addItem(&pixmap_item);
        _ui.simulation_graphicsView->setScene(&scene);

        reset_scale_view();
        initialize_gui_settings();
    });

    timer = new QTimer(parent);
    connect(timer, &QTimer::timeout, [&]{mainloop_tick();});

    set_window_interval(window_fps);
    set_simulation_interval(simulation_fps);

    timer->start();
    //reset_scale_view();
}

void WindowCore::mainloop_tick() {
    start = clock_now();
    if (synchronise_simulation_and_window) {
        cp.engine_pass_tick = true;
        cp.synchronise_simulation_tick = true;
    }
    window_tick();
    window_frames++;
    end = clock_now();
    delta_window_processing_time = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    // timer
    if (std::chrono::duration_cast<std::chrono::milliseconds>(clock_now() - fps_timer).count() / 1000. > 1) {
        //TODO make pausing logic better.
        //pauses engine, parses/loads data from/to engine, resumes engine. It looks stupid, but works
        cp.engine_pause = true;
        wait_for_engine_to_pause();
        simulation_frames = dc.engine_ticks;
        dc.engine_ticks = 0;
        unpause_engine();

        auto info = calculate_organisms_info();

        update_fps_labels(window_frames, simulation_frames);
        update_statistics_info(info);

        if (!stop_console_output) {
            std::cout << window_frames << " frames in a second | " << simulation_frames << " simulation ticks in second | "
            << dc.organisms.size() << " organisms alive | " << to_str(info.size, float_precision) << " average organism_size | "
            << to_str(dc.total_engine_ticks, float_precision) << " total engine ticks\n"
            << to_str(info.anatomy_mutation_rate, float_precision) << " average anatomy mutation rate | "
            << to_str(info.brain_mutation_rate, float_precision) << " average brain mutation rate | "
            << to_str(info._eye_blocks,      float_precision) << " average mouth num | "
            << to_str(info._producer_blocks, float_precision) << " average producer num | "
            << to_str(info._mover_blocks,    float_precision) << " average mover num | "
            << to_str(info._killer_blocks,   float_precision) << " average killer num | "
            << to_str(info._armor_blocks,    float_precision) << " average armor num | "
            << to_str(info._eye_blocks,      float_precision) << " average eye num\n"
            <<"\n";
        }

        window_frames = 0;
        fps_timer = clock_now();
    }
}

void WindowCore::update_fps_labels(int fps, int sps) {
    _ui.lb_fps->setText(QString::fromStdString("fps: " + std::to_string(fps)));
    _ui.lb_sps->setText(QString::fromStdString("sps: "+std::to_string(sps)));
}

//TODO kinda redundant right now
void WindowCore::window_tick() {
    if (resize_simulation_grid_flag) {resize_simulation_space(); resize_simulation_grid_flag=false;}
    if (sp.pause_on_total_extinction && cp.organisms_extinct) {_ui.tb_pause->setChecked(true); cp.organisms_extinct = false;} else
    if (sp.reset_on_total_extinction && cp.organisms_extinct) {
        reset_world();
        auto_reset_num++;
        _ui.lb_auto_reset_count->setText(QString::fromStdString("Auto reset count: "+ std::to_string(auto_reset_num)));
    }
    create_image();
}

void WindowCore::resize_image() {
    image_vector.clear();
    image_vector.reserve(4 * _ui.simulation_graphicsView->viewport()->width() * _ui.simulation_graphicsView->viewport()->height());
}

void WindowCore::move_center(int delta_x, int delta_y) {
    center_x -= delta_x * scaling_zoom;
    center_y -= delta_y * scaling_zoom;
}

void WindowCore::reset_scale_view() {
    center_x = (float)dc.simulation_width/2;
    center_y = (float)dc.simulation_height/2;
    // finds exponent needed to scale the sprite
    auto exp = log((float)dc.simulation_height/(float)_ui.simulation_graphicsView->viewport()->height()) / log(scaling_coefficient);
//    auto exp = log(1) / log(1.05);
    scaling_zoom = pow(scaling_coefficient, exp);
}

// TODO it is probably possible to do better.
//__attribute__((noinline))
QColor inline &WindowCore::get_color(BlockTypes type) {
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
    auto image_width = _ui.simulation_graphicsView->viewport()->width();
    auto image_height = _ui.simulation_graphicsView->viewport()->height();

    int scaled_width = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and stop coordinates on simulation grid
    auto start_x = (int)(center_x-(scaled_width / 2));
    auto end_x = (int)(center_x+(scaled_width / 2));

    auto start_y = (int)(center_y-(scaled_height / 2));
    auto end_y = (int)(center_y+(scaled_height / 2));

    std::vector<int> lin_width;
    std::vector<int> lin_height;

    calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    std::vector<int> truncated_lin_width; truncated_lin_width.reserve(image_width);
    std::vector<int> truncated_lin_height; truncated_lin_height.reserve(image_height);

    calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);

    if ((!pause_image_construction && !cp.engine_global_pause) || synchronise_simulation_and_window) {
        //it does not help
        //#pragma omp parallel for
        cp.engine_pause = true;
        // pausing engine to parse data from engine.
        auto paused = wait_for_engine_to_pause();
        // if for some reason engine is not paused in time, it will use old parsed data and not switch engine on.
        if (paused) {parse_simulation_grid(truncated_lin_width, truncated_lin_height); unpause_engine();}
    }

    image_for_loop(image_width, image_height, lin_width, lin_height);
    pixmap_item.setPixmap(QPixmap::fromImage(QImage(image_vector.data(), image_width, image_height, QImage::Format_RGB32)));
}

void inline WindowCore::calculate_linspace(std::vector<int> & lin_width, std::vector<int> & lin_height,
                                           int start_x, int end_x, int start_y, int end_y, int image_width, int image_height) {
    lin_width = Linspace<int>()(start_x, end_x, image_width);
    lin_height = Linspace<int>()(start_y, end_y, image_height);
}

void inline WindowCore::calculate_truncated_linspace(
        int image_width, int image_height,
        std::vector<int> & lin_width,
        std::vector<int> & lin_height,
        std::vector<int> & truncated_lin_width,
        std::vector<int> & truncated_lin_height) {

    int min_val = -1;
    for (int x = 0; x < image_width; x++) {if (lin_width[x] > min_val) {min_val = lin_width[x]; truncated_lin_width.push_back(min_val);}}
    min_val = -1;
    for (int y = 0; y < image_height; y++) {if (lin_height[y] > min_val) {min_val = lin_height[y]; truncated_lin_height.push_back(min_val);}}
}

//__attribute__((noinline))
void inline WindowCore::image_for_loop(int image_width, int image_height,
                                       std::vector<int> & lin_width,
                                       std::vector<int> & lin_height) {
    QColor pixel_color;
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            //TODO maybe rewrite in OpenGL?
            if (lin_width[x] < 0 || lin_width[x] >= dc.simulation_width || lin_height[y] < 0 || lin_height[y] >= dc.simulation_height) {pixel_color = color_container.simulation_background_color;}
            else {pixel_color = get_color(dc.second_simulation_grid[lin_width[x]][lin_height[y]].type);}
            set_image_pixel(x, y, pixel_color);
        }
    }
}

// depth * ( y * width + x) + z
// depth * width * y + depth * x + z
void WindowCore::set_image_pixel(int x, int y, QColor &color) {
    auto index = 4 * (y * _ui.simulation_graphicsView->viewport()->width() + x);
    image_vector[index+2] = color.red();
    image_vector[index+1] = color.green();
    image_vector[index  ] = color.blue();
}

bool WindowCore::compare_pixel_color(int x, int y, QColor &color) {
    auto index = 4 * (y * _ui.simulation_graphicsView->viewport()->width() + x);
    return image_vector[index+2] == color.red()  &&
           image_vector[index+1] == color.green()&&
           image_vector[index  ] == color.blue();
}

void WindowCore::set_window_interval(int max_window_fps) {
    if (max_window_fps <= 0) {
        window_interval = 0.;
        timer->setInterval(0);
        return;
    }
    window_interval = 1./max_window_fps;
    timer->setInterval(1000/max_window_fps);
}

void WindowCore::set_simulation_interval(int max_simulation_fps) {
    if (max_simulation_fps <= 0) {
        dc.simulation_interval = 0.;
        dc.unlimited_simulation_fps = true;
        return;
    }
    dc.simulation_interval = 1./max_simulation_fps;
    dc.unlimited_simulation_fps = false;
}

__attribute__((optimize("O0")))
bool WindowCore::wait_for_engine_to_pause() {
//    // if engine is paused by user, then return if engine is really paused.
//    if (cp.engine_global_pause && !synchronise_simulation_and_window) {return cp.engine_paused;}
//    auto sleeping_time = std::chrono::microseconds (int(dc.delta_time*1.5));
//    // if sleeping time is larger than spare window processing time, then not wait and return result straight away.
//    //TODO not needed
//    if (!allow_waiting_overhead) {
//        if (sleeping_time.count() > int(window_interval * std::chrono::microseconds::period::den) - delta_window_processing_time) { return cp.engine_paused; }
//    } else {
//        if (sleeping_time.count() > int(window_interval * std::chrono::microseconds::period::den)) { return cp.engine_paused; }
//    }
//    std::this_thread::sleep_for(sleeping_time);
    auto now = clock_now();
    while (!cp.engine_paused) {
        if (!stop_console_output && std::chrono::duration_cast<std::chrono::milliseconds>(clock_now() - now).count() / 1000 > 1) {
            std::cout << "Waiting for engine to pause\n";
            now = clock_now();
        }
    }
    return cp.engine_paused;
}

void WindowCore::parse_simulation_grid(std::vector<int> & lin_width, std::vector<int> & lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= dc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= dc.simulation_height) { continue; }
            dc.second_simulation_grid[x][y].type = dc.single_thread_simulation_grid[x][y].type;
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
            dc.second_simulation_grid[x][y].type = dc.single_thread_simulation_grid[x][y].type;
        }
    }
    unpause_engine();
}

void WindowCore::set_simulation_num_threads(uint8_t num_threads) {
    cp.num_threads = num_threads;
    cp.build_threads = true;
}

void WindowCore::set_cursor_mode(CursorMode mode) {
    cursor_mode = mode;
}

void WindowCore::set_simulation_mode(SimulationModes mode) {
    cp.change_to_mode = mode;
    cp.change_simulation_mode = true;
}

void WindowCore::clear_organisms() {
    if (!cp.engine_paused) {
        if (!stop_console_output) {std::cout << "Engine is not paused! Organism not cleared.\n";}
        return;
    }
    for (auto & organism: dc.organisms) {delete organism;}
    for (auto & organism: dc.to_place_organisms) {delete organism;}
    dc.organisms.clear();
    dc.to_place_organisms.clear();
}

//TODO there could be memory leaks in the feature.
void WindowCore::resize_simulation_space() {
    auto msg = DescisionMessageBox("Warning",
                                   "Warning, resizing simulation grid is unstable and can crash the program.\n"
                                   "I don't know why it crashes, but beware that it can.",
                                   "OK", "Cancel", this);
    auto result = msg.exec();
    if (!result) {
        return;
    }

    cp.engine_pause = true;
    wait_for_engine_to_pause();

    dc.simulation_width = new_simulation_width;
    dc.simulation_height = new_simulation_height;

    dc.single_thread_simulation_grid.clear();
    dc.second_simulation_grid.clear();

    auto block = BaseGridBlock{};
    block.type = BlockTypes::EmptyBlock;

    dc.single_thread_simulation_grid.resize(dc.simulation_width, std::vector<BaseGridBlock>(dc.simulation_height, block));
    dc.second_simulation_grid.resize(dc.simulation_width, std::vector<BaseGridBlock>(dc.simulation_height, block));

    clear_organisms();

    make_walls();

    cp.build_threads = true;
    reset_world();

    unpause_engine();

    reset_scale_view();
}

void WindowCore::partial_clear_world() {
    cp.engine_pause = true;
    wait_for_engine_to_pause();

    clear_organisms();

    for (auto & column: dc.single_thread_simulation_grid)        {for (auto & block: column) { block.type = BlockTypes::EmptyBlock;}}
    for (auto & column: dc.second_simulation_grid) {for (auto & block: column) {block.type = BlockTypes::EmptyBlock;}}

    dc.total_engine_ticks = 0;
}

void WindowCore::reset_world() {
    partial_clear_world();
    clear_organisms();
    make_walls();

    base_organism->x = dc.simulation_width / 2;
    base_organism->y = dc.simulation_height / 2;

    chosen_organism->x = dc.simulation_width / 2;
    chosen_organism->y = dc.simulation_height / 2;

    if (reset_with_chosen) {dc.to_place_organisms.push_back(new Organism(chosen_organism));}
    else                   {dc.to_place_organisms.push_back(new Organism(base_organism));}
    reset_with_chosen = false;
    //Just in case
    cp.engine_pass_tick = true;
    cp.synchronise_simulation_tick = true;
    unpause_engine();
}

void WindowCore::clear_world() {
    partial_clear_world();
    make_walls();
    unpause_engine();
}

void WindowCore::unpause_engine() {
    if (!synchronise_simulation_and_window) {
        cp.engine_pause = false;
    }
}

void WindowCore::make_walls() {
    auto wall_thickness = 1;

    for (int x = 0; x < dc.simulation_width; x++) {
        for (int i = 0; i < wall_thickness; i++) {
            dc.single_thread_simulation_grid[x][i].type = BlockTypes::WallBlock;
            dc.single_thread_simulation_grid[x][dc.simulation_height - 1 - i].type = BlockTypes::WallBlock;
        }
    }
    for (int y = 0; y < dc.simulation_width; y++) {
        for (int i = 0; i < wall_thickness; i++) {
            dc.single_thread_simulation_grid[i][y].type = BlockTypes::WallBlock;
            dc.single_thread_simulation_grid[dc.simulation_width - 1 - i][y].type = BlockTypes::WallBlock;
        }
    }
}

OrganismAvgBlockInformation WindowCore::calculate_organisms_info() {
    OrganismAvgBlockInformation info;
    for (auto & organism: dc.organisms) {
        info.size += organism->organism_anatomy->_organism_blocks.size();

        info._mouth_blocks    += organism->organism_anatomy->_mouth_blocks;
        info._producer_blocks += organism->organism_anatomy->_producer_blocks;
        info._mover_blocks    += organism->organism_anatomy->_mover_blocks;
        info._killer_blocks   += organism->organism_anatomy->_killer_blocks;
        info._armor_blocks    += organism->organism_anatomy->_armor_blocks;
        info._eye_blocks      += organism->organism_anatomy->_eye_blocks;

        info.brain_mutation_rate   += organism->brain_mutation_rate;
        info.anatomy_mutation_rate += organism->anatomy_mutation_rate;
    }

    info.size /= dc.organisms.size();

    info._mouth_blocks    /= dc.organisms.size();
    info._producer_blocks /= dc.organisms.size();
    info._mover_blocks    /= dc.organisms.size();
    info._killer_blocks   /= dc.organisms.size();
    info._armor_blocks    /= dc.organisms.size();
    info._eye_blocks      /= dc.organisms.size();

    info.brain_mutation_rate   /= dc.organisms.size();
    info.anatomy_mutation_rate /= dc.organisms.size();

    if (std::isnan(info.size))             {info.size             = 0;}

    if (std::isnan(info._mouth_blocks))    {info._mouth_blocks    = 0;}
    if (std::isnan(info._producer_blocks)) {info._producer_blocks = 0;}
    if (std::isnan(info._mover_blocks))    {info._mover_blocks    = 0;}
    if (std::isnan(info._killer_blocks))   {info._killer_blocks   = 0;}
    if (std::isnan(info._armor_blocks))    {info._armor_blocks    = 0;}
    if (std::isnan(info._eye_blocks))      {info._eye_blocks      = 0;}

    if (std::isnan(info.brain_mutation_rate)) {info.brain_mutation_rate    = 0;}
    if (std::isnan(info.anatomy_mutation_rate)) {info.anatomy_mutation_rate      = 0;}
    return info;
}

void WindowCore::update_statistics_info(OrganismAvgBlockInformation info) {
    _ui.lb_total_engine_ticks->setText(QString::fromStdString("Total engine ticks: "    + std::to_string(dc.total_engine_ticks)));
    _ui.lb_organisms_alive->   setText(QString::fromStdString("Organism alive: "        + std::to_string(dc.organisms.size())));
    _ui.lb_organism_size->     setText(QString::fromStdString("Average organism size: " + to_str(info.size,             float_precision)));
    _ui.lb_mouth_num->         setText(QString::fromStdString("Average mouth num: "     + to_str(info._mouth_blocks,    float_precision)));
    _ui.lb_producer_num->      setText(QString::fromStdString("Average producer num: "  + to_str(info._producer_blocks, float_precision)));
    _ui.lb_mover_num->         setText(QString::fromStdString("Average mover num: "     + to_str(info._mover_blocks,    float_precision)));
    _ui.lb_killer_num->        setText(QString::fromStdString("Average killer num: "    + to_str(info._killer_blocks,   float_precision)));
    _ui.lb_armor_num->         setText(QString::fromStdString("Average armor num: "     + to_str(info._armor_blocks,    float_precision)));
    _ui.lb_eye_num->           setText(QString::fromStdString("Average eye num: "       + to_str(info._eye_blocks,      float_precision)));

    _ui.lb_anatomy_mutation_rate->setText(QString::fromStdString("Average anatomy mutation rate: " + to_str(info.anatomy_mutation_rate, float_precision)));
    _ui.lb_brain_mutation_rate->  setText(QString::fromStdString("Average brain mutation rate: "   + to_str(info.brain_mutation_rate,   float_precision)));
}

// So that changes in code values would be set by default in gui.
void WindowCore::initialize_gui_settings() {
    //World settings
    _ui.le_cell_size->setText(QString::fromStdString(std::to_string(cell_size)));
    _ui.le_simulation_width->setText(QString::fromStdString(std::to_string(dc.simulation_width)));
    _ui.le_simulation_height->setText(QString::fromStdString(std::to_string(dc.simulation_height)));
    _ui.cb_reset_on_total_extinction->setChecked(sp.reset_on_total_extinction);
    _ui.cb_pause_on_total_extinction->setChecked(sp.pause_on_total_extinction);
    _ui.le_max_organisms->setText(QString::fromStdString(std::to_string(dc.max_organisms)));
    _ui.cb_fill_window->setChecked(fill_window);
    //Evolution settings
    _ui.le_food_production_probability->setText(QString::fromStdString(to_str(sp.food_production_probability, float_precision)));
    _ui.le_produce_food_every_n_tick->setText(QString::fromStdString(std::to_string(sp.produce_food_every_n_life_ticks)));
    _ui.le_lifespan_multiplier->setText(QString::fromStdString(std::to_string(sp.lifespan_multiplier)));
    _ui.le_look_range->setText(QString::fromStdString(std::to_string(sp.look_range)));
    _ui.le_auto_food_drop_rate->setText(QString::fromStdString(std::to_string(sp.auto_food_drop_rate)));
    _ui.le_extra_reproduction_cost->setText(QString::fromStdString(std::to_string(sp.extra_reproduction_cost)));
    _ui.cb_use_evolved_anatomy_mutation_rate->setChecked(sp.use_anatomy_evolved_mutation_rate);
    _ui.le_global_anatomy_mutation_rate->setText(QString::fromStdString(to_str(sp.global_anatomy_mutation_rate, float_precision)));
    _ui.cb_use_evolved_brain_mutation_rate->setChecked(sp.use_brain_evolved_mutation_rate);
    _ui.le_global_brain_mutation_rate->setText(QString::fromStdString(to_str(sp.global_brain_mutation_rate, float_precision)));
    _ui.le_add->setText(QString::fromStdString(std::to_string(sp.add_cell)));
    _ui.le_change->setText(QString::fromStdString(std::to_string(sp.change_cell)));
    _ui.le_remove->setText(QString::fromStdString(std::to_string(sp.remove_cell)));
    _ui.cb_reproducing_rotation_enabled->setChecked(sp.reproduction_rotation_enabled);
    _ui.cb_runtime_rotation_enabled->setChecked(sp.runtime_rotation_enabled);
    _ui.cb_on_touch_kill->setChecked(sp.on_touch_kill);
    _ui.cb_movers_can_produce_food->setChecked(sp.movers_can_produce_food);
    _ui.cb_food_blocks_reproduction->setChecked(sp.food_blocks_reproduction);
    _ui.le_min_reproduction_distance->setText(QString::fromStdString(std::to_string(sp.min_reproducing_distance)));
    _ui.le_max_reproduction_distance->setText(QString::fromStdString(std::to_string(sp.max_reproducing_distance)));
    _ui.cb_fix_reproduction_distance->setChecked(sp.reproduction_distance_fixed);
    //Simulation settings
    _ui.cb_stop_console_output->setChecked(stop_console_output);
    _ui.le_num_threads->setText(QString::fromStdString(std::to_string(cp.num_threads)));
    _ui.le_float_number_precision->setText(QString::fromStdString(std::to_string(float_precision)));
}