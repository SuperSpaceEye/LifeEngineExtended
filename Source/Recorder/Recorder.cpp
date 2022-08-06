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
