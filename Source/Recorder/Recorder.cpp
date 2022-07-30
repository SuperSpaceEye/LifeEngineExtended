//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

Recorder::Recorder(Ui::MainWindow *_parent_ui, EngineDataContainer * edc, EngineControlParameters * ecp, ColorContainer * cc, Textures * textures,
                   RecordingData * recording_data):
    parent_ui(_parent_ui), edc(edc), ecp(ecp), cc(cc), textures(textures), recd(recording_data) {
    _ui.setupUi(this);
    init_gui();
}

void Recorder::set_engine(SimulationEngine * engine) {
    this->engine = engine;
}

void Recorder::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_recorder_window->setChecked(false);
    QWidget::closeEvent(event);
}

OrganismAvgBlockInformation Recorder::parse_organisms_info() {
    OrganismAvgBlockInformation info;

    bool has_pool = true;
    int i = 0;
    //Why while loop? the easier implementation with for loop randomly crashes sometimes, and I don't know why.
    while (has_pool) {
        std::vector<Organism*> * pool;

        if (ecp->simulation_mode == SimulationModes::CPU_Single_Threaded) {
            pool = &edc->organisms;
            has_pool = false;
        } else if (ecp->simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
            pool = &edc->organisms_pools[i];
            i++;
            if (i >= ecp->num_threads) {
                has_pool = false;
            }
        } else {
            throw "no pool";
        }

        for (auto & organism: *pool) {
            info.total_size_organism_blocks += organism->anatomy._organism_blocks.size();
            info.total_size_producing_space += organism->anatomy._producing_space.size();
            info.total_size_eating_space    += organism->anatomy._eating_space.size();

            if (organism->anatomy._mover_blocks > 0) {
                info.move_range += organism->move_range;
                info.moving_organisms++;

                if (organism->anatomy._eye_blocks > 0) {
                    info.organisms_with_eyes++;
                }
            }

            info.total_avg.size += organism->anatomy._organism_blocks.size();

            info.total_avg._organism_lifetime += organism->max_lifetime;
            info.total_avg._organism_age      += organism->lifetime;
            info.total_avg._mouth_blocks      += organism->anatomy._mouth_blocks;
            info.total_avg._producer_blocks   += organism->anatomy._producer_blocks;
            info.total_avg._mover_blocks      += organism->anatomy._mover_blocks;
            info.total_avg._killer_blocks     += organism->anatomy._killer_blocks;
            info.total_avg._armor_blocks      += organism->anatomy._armor_blocks;
            info.total_avg._eye_blocks        += organism->anatomy._eye_blocks;

            info.total_avg.brain_mutation_rate   += organism->brain_mutation_rate;
            info.total_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
            info.total_avg.total++;

            if (organism->anatomy._mover_blocks > 0) {
                info.moving_avg.size += organism->anatomy._organism_blocks.size();

                info.moving_avg._organism_lifetime += organism->max_lifetime;
                info.moving_avg._organism_age      += organism->lifetime;
                info.moving_avg._mouth_blocks      += organism->anatomy._mouth_blocks;
                info.moving_avg._producer_blocks   += organism->anatomy._producer_blocks;
                info.moving_avg._mover_blocks      += organism->anatomy._mover_blocks;
                info.moving_avg._killer_blocks     += organism->anatomy._killer_blocks;
                info.moving_avg._armor_blocks      += organism->anatomy._armor_blocks;
                info.moving_avg._eye_blocks        += organism->anatomy._eye_blocks;

                info.moving_avg.brain_mutation_rate   += organism->brain_mutation_rate;
                info.moving_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
                info.moving_avg.total++;
            } else {
                info.station_avg.size += organism->anatomy._organism_blocks.size();

                info.station_avg._organism_lifetime += organism->max_lifetime;
                info.station_avg._organism_age      += organism->lifetime;
                info.station_avg._mouth_blocks      += organism->anatomy._mouth_blocks;
                info.station_avg._producer_blocks   += organism->anatomy._producer_blocks;
                info.station_avg._mover_blocks      += organism->anatomy._mover_blocks;
                info.station_avg._killer_blocks     += organism->anatomy._killer_blocks;
                info.station_avg._armor_blocks      += organism->anatomy._armor_blocks;
                info.station_avg._eye_blocks        += organism->anatomy._eye_blocks;

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
                      (sizeof(Organism) * info.total_avg.total);

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

void Recorder::create_image(std::vector<unsigned char> &raw_image_data, std::vector<BaseGridBlock> &grid,
                            int simulation_width, int simulation_height, int num_pixels_per_block) {
    auto image_width  = simulation_width  * num_pixels_per_block;
    auto image_height = simulation_height * num_pixels_per_block;

    // start and y coordinates on simulation grid
    auto start_x = 0;
    auto end_x = simulation_width;

    auto start_y = 0;
    auto end_y = simulation_height;

    std::vector<int> lin_width;
    std::vector<int> lin_height;

    lin_width  = linspace<int>(start_x, end_x, image_width);
    lin_height = linspace<int>(start_y, end_y, image_height);

    int max_x = lin_width[lin_width.size()-1];
    int max_y = lin_height[lin_height.size()-1];
    int del_x = 0;
    int del_y = 0;
    for (int x = lin_width.size() -1; lin_width[x]  == max_x; x--) {del_x++;}
    for (int y = lin_height.size()-1; lin_height[y] == max_y; y--) {del_y++;}

    for (int i = 0; i < del_x; i++) {lin_width.pop_back();}
    for (int i = 0; i < del_y; i++) {lin_height.pop_back();}

    std::vector<int> truncated_lin_width;
    truncated_lin_width.reserve(image_width);
    std::vector<int> truncated_lin_height;
    truncated_lin_height.reserve(image_height);

    int min_val = INT32_MIN;
    for (int x = 0; x < image_width; x++) {if (lin_width[x] > min_val) {min_val = lin_width[x]; truncated_lin_width.push_back(min_val);}}
    truncated_lin_width.pop_back();
    min_val = INT32_MIN;
    for (int y = 0; y < image_height; y++) {if (lin_height[y] > min_val) {min_val = lin_height[y]; truncated_lin_height.push_back(min_val);}}
    truncated_lin_height.pop_back();

    complex_image_creation(lin_width, lin_height, raw_image_data, grid);
}

void Recorder::complex_image_creation(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                                      std::vector<unsigned char> &raw_image_vector, std::vector<BaseGridBlock> &grid) {
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
                    auto &block = grid[lin_width[x] + lin_height[y] * edc->simulation_width];

                    if (lin_width[x] < 0 ||
                        lin_width[x] >= edc->simulation_width ||
                        lin_height[y] < 0 ||
                        lin_height[y] >= edc->simulation_height) {
                        pixel_color = cc->simulation_background_color;
                    } else {
                        pixel_color = get_texture_color(block.type,
                                                        block.rotation,
                                                        float(x - w_b.x) / (w_b.y - w_b.x),
                                                        float(y - h_b.x) / (h_b.y - h_b.x));
                    }
                    set_image_pixel(x, y, pixel_color, raw_image_vector);
                }
            }
        }
    }
}

// depth * ( y * width + x) + z
// depth * width * y + depth * x + z
void Recorder::set_image_pixel(int x, int y, const color &color, std::vector<unsigned char> &image_vector) {
    auto index = 4 * (y * edc->simulation_width * num_pixels_per_block + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

color & Recorder::get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale) {
    int x;
    int y;

    switch (type) {
        case BlockTypes::EmptyBlock :   return cc->empty_block;
        case BlockTypes::MouthBlock:    return cc->mouth;
        case BlockTypes::ProducerBlock: return cc->producer;
        case BlockTypes::MoverBlock:    return cc->mover;
        case BlockTypes::KillerBlock:   return cc->killer;
        case BlockTypes::ArmorBlock:    return cc->armor;
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
            return textures->rawEyeTexture[x + y * 5];
        case BlockTypes::FoodBlock:     return cc->food;
        case BlockTypes::WallBlock:     return cc->wall;
        default: return cc->empty_block;
    }
}

void Recorder::init_gui() {
    _ui.le_number_or_pixels_per_block->setText(QString::fromStdString(std::to_string(num_pixels_per_block)));
    _ui.le_log_every_n_tick->setText(QString::fromStdString(std::to_string(ecp->parse_full_grid_every_n)));
    _ui.le_first_grid_buffer_size->setText(QString::fromStdString(std::to_string(recd->buffer_size)));
    _ui.le_video_fps->setText(QString::fromStdString(std::to_string(video_fps)));
}

std::string Recorder::new_recording(std::string path) {
    auto new_path = path + "/temp/" + get_string_date();
    std::filesystem::create_directory(new_path);
    return new_path;
}

std::string Recorder::get_string_date() {
    time_t t = time(nullptr);
    auto my_time = localtime(&t);
    std::string date = std::to_string(my_time->tm_year + 1900) +
                 "_" + std::to_string(my_time->tm_mon+1) +
                 "_" + std::to_string(my_time->tm_mday)  +
                 "_" + std::to_string(my_time->tm_hour)  +
                 "_" + std::to_string(my_time->tm_min)   +
                 "_" + std::to_string(my_time->tm_sec);
    return date;
}

void Recorder::update_label() {
    std::string status;
    if (ecp->record_full_grid) {
        status = "Recording";
    } else {
        if (recording_paused) {
            status = "Paused";
        } else {
            if (recd->path_to_save.empty()) {
                status = "No recording";
            } else {
                status = "Stopped";
            }
        }
    }
    std::string rec_states = std::to_string(recd->recorded_states);
    std::string buffer_filling = std::to_string(recd->buffer_pos) + "/" + std::to_string(recd->buffer_size);
    std::string size_of_recording = "0 B";
    if (ecp->record_full_grid || recording_paused) {
        uint64_t size = 0;
        for (auto & entry: std::filesystem::directory_iterator(recd->path_to_save)) {
            size += entry.file_size();
        }
        size_of_recording = convert_num_bytes(size);
    }
    std::string time_length_of_recording = convert_seconds(recd->recorded_states / video_fps);
    std::string str = "Status: " + status + " ||| Recorded " + rec_states + " ticks ||| Buffer filling: " + buffer_filling  + " ||| Recording size on disk: " + size_of_recording + " ||| Time length of recording with " + std::to_string(video_fps) + " fps: " + time_length_of_recording;
    _ui.lb_recording_information->setText(QString::fromStdString(str));
}

void Recorder::clear_data() {
    recd->buffer_pos = 0;
    recd->recorded_states = 0;
    recd->path_to_save = "";
    recording_paused = false;

    recd->second_simulation_grid_buffer.clear();
    recd->second_simulation_grid_buffer.shrink_to_fit();
}






