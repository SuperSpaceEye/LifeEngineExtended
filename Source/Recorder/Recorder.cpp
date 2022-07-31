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

    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    ImageCreation::ImageCreationTools::complex_image_creation(lin_width,
                                                              lin_height,
                                                              edc->simulation_width,
                                                              edc->simulation_height,
                                                              *cc,
                                                              *textures,
                                                              image_width,
                                                              raw_image_data,
                                                              grid);
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






