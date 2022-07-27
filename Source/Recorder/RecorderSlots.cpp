//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

//ffmpeg -framerate 60 -pattern_type glob -i "../images/*.png" -c:v libx264 out.mp4 -y

void Recorder::le_number_of_pixels_per_block_slot() {
    le_slot_lower_bound<int>(num_pixels_per_block, num_pixels_per_block, "int", _ui.le_number_or_pixels_per_block, 1, "1");
}

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