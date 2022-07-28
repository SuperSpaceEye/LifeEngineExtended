//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

//ffmpeg -framerate 60 -pattern_type glob -i "../images/*.png" -c:v libx264 out.mp4 -y

//==================== Line edits ====================

void Recorder::le_number_of_pixels_per_block_slot() {
    le_slot_lower_bound<int>(num_pixels_per_block, num_pixels_per_block, "int", _ui.le_number_or_pixels_per_block, 1, "1");
}

void Recorder::le_first_grid_buffer_size_slot() {

}

void Recorder::le_second_grid_buffer_size_slot() {

}

//==================== Buttons edits ====================

void Recorder::b_create_image_slot() {
    engine->pause();

    bool flag = ecp->synchronise_simulation_and_window;
    ecp->synchronise_simulation_and_window = false;

    QString selected_filter;
    QFileDialog file_dialog{};

    auto file_name = file_dialog.getSaveFileName(this, tr("Save Image"), "",
                                                 "PNG (*.png)", &selected_filter);
#ifndef __WIN32
    bool file_exists = std::filesystem::exists(file_name.toStdString());
#endif

    std::string filetype;
    if (selected_filter.toStdString() == "PNG (*.png)") {
        filetype = ".png";
    } else {
        ecp->synchronise_simulation_and_window = flag;
        engine->unpause();
        return;
    }
    std::string full_path = file_name.toStdString();

#ifndef __WIN32
    if (!file_exists) {
        full_path = file_name.toStdString() + filetype;
    }
#endif

    std::vector<unsigned char> raw_image_data(edc->simulation_width*edc->simulation_height*num_pixels_per_block*num_pixels_per_block*4);

    create_image(raw_image_data);

    QImage image(raw_image_data.data(),
                 edc->simulation_width*num_pixels_per_block,
                 edc->simulation_height*num_pixels_per_block,
                 QImage::Format_RGB32);

    image.save(QString::fromStdString(full_path), "PNG");

    ecp->synchronise_simulation_and_window = flag;
    engine->unpause();
}

void Recorder::b_start_recording_slot(){}
void Recorder::b_stop_recording_slot(){}
void Recorder::b_load_intermediate_data_location_slot() {
    if (ecp->parse_full_grid) {
        display_message("Program is still recording. Stop the recording first to load intermediate data.");
        return;
    }
    QFileDialog file_dialog{};
    auto dir = QCoreApplication::applicationDirPath() + "/temp";

    auto dir_name = file_dialog.getExistingDirectory(this, tr("Open directory"), dir);
    if (dir_name.toStdString().empty()) {
        return;
    }

    edc->second_simulation_grid_buffer.clear();
    second_buffer.clear();

    dir_path_of_new_recording = dir_name.toStdString();
}

void Recorder::b_compile_intermediate_data_into_video_slot(){}

void Recorder::b_clear_intermediate_data_slot() {
    if (ecp->parse_full_grid) {
        display_message("Program is still recording. Stop the recording first to clear intermediate data.");
        return;
    }

    edc->second_simulation_grid_buffer.clear();
    second_buffer.clear();
}

void Recorder::b_delete_all_intermediate_data_from_disk_slot() {
    if (ecp->parse_full_grid) {
        display_message("Program is still recording. Stop the recording first to delete intermediate data.");
        return;
    }
    edc->second_simulation_grid_buffer.clear();
    second_buffer.clear();

    auto path = QCoreApplication::applicationDirPath().toStdString() + "/temp";

    std::vector<std::string> subdirectories;
    for (const auto & entry: std::filesystem::directory_iterator(path)) {
        subdirectories.emplace_back(entry.path());
    }

    for (auto & dir: subdirectories) {
        std::filesystem::remove_all(dir);
    }
}

void Recorder::b_new_recording_slot() {
    auto path = QCoreApplication::applicationDirPath().toStdString();
    dir_path_of_new_recording = new_recording(path);
}