//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

Recorder::Recorder(Ui::MainWindow *_parent_ui, EngineDataContainer *edc, EngineControlParameters *ecp,
                   ColorContainer *cc,
                   TexturesContainer *textures, TransactionBuffer *tbuffer, float *viewpoint_x, float *viewpoint_y,
                   float *zoom, const bool &cuda_is_available) :
    parent_ui(_parent_ui), edc(edc), ecp(ecp), cc(cc), textures(textures), tbuffer(tbuffer), main_viewpoint_x(viewpoint_x),
    main_viewpoint_y(viewpoint_y), main_zoom(zoom), cuda_is_available(cuda_is_available){
    ui.setupUi(this);
    init_gui();
}

void Recorder::set_engine(SimulationEngine * engine) {
    this->engine = engine;
}

void Recorder::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_recorder_window->setChecked(false);
    QWidget::closeEvent(event);
}

void Recorder::create_image(std::vector<unsigned char> &raw_image_data, const std::vector<BaseGridBlock> &grid,
                            int simulation_width, int simulation_height, int num_pixels_per_block, bool use_cuda,
                            bool use_viewpoint) {
    int image_width;
    int image_height;
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    std::vector<int> lin_width;
    std::vector<int> lin_height;
    std::vector<int> truncated_lin_width;
    std::vector<int> truncated_lin_height;

    if (!use_viewpoint) {
        prepare_full_view(simulation_width, simulation_height, num_pixels_per_block, image_height, start_x, end_x,
                          start_y,
                          end_y, truncated_lin_width, truncated_lin_height, image_width, lin_width, lin_height);
    } else {
        prepare_relative_view(lin_height, truncated_lin_width,
                              truncated_lin_height, image_width, image_height, start_x, end_x, start_y, end_y,
                              lin_width);
    }

#if __CUDA_USED__
    void * cuda_i_creator = &cuda_image_creator;
#else
    void * cuda_i_creator = nullptr;
#endif

    ImageCreation::create_image(lin_width, lin_height, edc->simulation_width, edc->simulation_height,
                                *cc, *textures, image_width, image_height, raw_image_data, grid, use_cuda, cuda_is_available,
                                cuda_i_creator, truncated_lin_width, truncated_lin_height);
}

void
Recorder::prepare_full_view(int simulation_width, int simulation_height, int num_pixels_per_block, int &image_height,
                            int start_x, int end_x, int start_y, int end_y, std::vector<int> &truncated_lin_width,
                            std::vector<int> &truncated_lin_height, int &image_width, std::vector<int> &lin_width,
                            std::vector<int> &lin_height) const {
    image_width  = simulation_width * num_pixels_per_block;
    image_height = simulation_height * num_pixels_per_block;

    start_x= 0;
    end_x= simulation_width;
    start_y= 0;
    end_y= simulation_height;

    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    truncated_lin_width .reserve(abs(lin_width [lin_width.size() - 1]) + 1);
    truncated_lin_height.reserve(abs(lin_height[lin_height.size() - 1]) + 1);

    ImageCreation::calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);
}

void Recorder::prepare_relative_view(std::vector<int> &lin_height, std::vector<int> &truncated_lin_width,
                                     std::vector<int> &truncated_lin_height, int &image_width, int &image_height,
                                     int &start_x, int &end_x, int &start_y, int &end_y,
                                     std::vector<int> &lin_width) const {
    image_width  = this->image_width;
    image_height = this->image_height;

    start_x = int(viewpoint_x - (image_width * zoom) / 2);
    end_x   = int(viewpoint_x + (image_width * zoom) / 2);

    start_y = int(viewpoint_y - (image_height * zoom) / 2);
    end_y   = int(viewpoint_y + (image_height * zoom) / 2);

    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    truncated_lin_width .reserve(abs(lin_width [lin_width.size() - 1]) + 1);
    truncated_lin_height.reserve(abs(lin_height[lin_height.size() - 1]) + 1);

    ImageCreation::calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);
}

void Recorder::init_gui() {
    ui.le_number_or_pixels_per_block->setText(QString::fromStdString(std::to_string(num_pixels_per_block)));
    ui.le_log_every_n_tick->setText(QString::fromStdString(std::to_string(ecp->parse_full_grid_every_n)));
    ui.le_first_grid_buffer_size->setText(QString::fromStdString(std::to_string(tbuffer->buffer_size)));
    ui.le_video_fps->setText(QString::fromStdString(std::to_string(video_fps)));
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
    if (edc->record_data) {
        status = "Recording";
    } else {
        if (recording_paused) {
            status = "Paused";
        } else {
            if (tbuffer->path_to_save.empty()) {
                status = "No recording";
            } else {
                status = "Stopped";
            }
        }
    }
    std::string rec_states = std::to_string(tbuffer->recorded_transactions);
    std::string rec_video_states = std::to_string(int(tbuffer->recorded_transactions/ecp->parse_full_grid_every_n));
    std::string buffer_filling = std::to_string(tbuffer->buffer_pos) + "/" + std::to_string(tbuffer->buffer_size);
    std::string size_of_recording = "0 B";
    if (edc->record_data || recording_paused) {
        uint64_t size = 0;
        if (!tbuffer->path_to_save.empty()) {
            for (auto &entry: std::filesystem::directory_iterator(tbuffer->path_to_save)) {
                size += entry.file_size();
            }
        }
        size_of_recording = convert_num_bytes(size);
    }
    std::string time_length_of_recording = convert_seconds(tbuffer->recorded_transactions / ecp->parse_full_grid_every_n / video_fps);
    std::string str = "Status: " + status + " ||| Recorded " + rec_states + " ticks ||| Video ticks " + rec_video_states  + " ||| Buffer filling: " + buffer_filling  + " ||| Recording size on disk: " + size_of_recording + " ||| Time length of recording with " + std::to_string(video_fps) + " fps: " + time_length_of_recording;
    ui.lb_recording_information->setText(QString::fromStdString(str));
}

void Recorder::clear_data() {
    tbuffer->finish_recording();
}
