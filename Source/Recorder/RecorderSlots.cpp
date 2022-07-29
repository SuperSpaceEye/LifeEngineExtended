//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

//==================== Line edits ====================

void Recorder::le_number_of_pixels_per_block_slot() {
    le_slot_lower_bound<int>(num_pixels_per_block, num_pixels_per_block, "int", _ui.le_number_or_pixels_per_block, 1, "1");
}

void Recorder::le_first_grid_buffer_size_slot() {
    int temp;
    le_slot_lower_bound<int>(temp, temp, "int", _ui.le_first_grid_buffer_size, 1, "1");

    if (temp == recd->buffer_pos) { return;}

    ecp->pause_buffer_filling = true;
    while (ecp->recording_full_grid) {}

    if (temp <= recd->buffer_pos) {
        recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc->simulation_width, edc->simulation_height, recd->second_simulation_grid_buffer);
    }
    recd->buffer_pos = 0;
    recd->buffer_size = temp;
    recd->second_simulation_grid_buffer.resize(recd->buffer_size, std::vector<BaseGridBlock>(edc->simulation_height*edc->simulation_width));
    recd->second_simulation_grid_buffer.shrink_to_fit();
    ecp->pause_buffer_filling = false;
}

void Recorder::le_log_every_n_tick_slot() {
    int temp = ecp->parse_full_grid_every_n;
    le_slot_lower_bound<int>(temp, temp, "int", _ui.le_log_every_n_tick, 1, "1");
    ecp->parse_full_grid_every_n = temp;
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

    engine->parse_full_simulation_grid();

    create_image(raw_image_data, edc->second_simulation_grid);

    QImage image(raw_image_data.data(),
                 edc->simulation_width*num_pixels_per_block,
                 edc->simulation_height*num_pixels_per_block,
                 QImage::Format_RGB32);

    image.save(QString::fromStdString(full_path), "PNG");

    ecp->synchronise_simulation_and_window = flag;
    engine->unpause();
}

void Recorder::b_start_recording_slot() {
    //if already recording
    if (ecp->record_full_grid) { return;}
    if (recd->path_to_save.empty()) {
        display_message("Make new recording to start.");
        return;
    }

    if (!recording_paused) {
        recd->buffer_pos = 0;
        recd->recorded_states = 0;
        recd->saved_buffers = 0;
        recd->second_simulation_grid_buffer.resize(recd->buffer_size, std::vector<BaseGridBlock>(edc->simulation_height*edc->simulation_width));
    }

    recording_paused = false;
    ecp->record_full_grid = true;
}

void Recorder::b_stop_recording_slot() {
    ecp->record_full_grid = false;
    while (ecp->recording_full_grid) {}

    recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc->simulation_width, edc->simulation_height, recd->second_simulation_grid_buffer);

    b_clear_intermediate_data_slot();
}

void Recorder::b_pause_recording_slot() {
    ecp->record_full_grid = false;
    recording_paused = true;
    while (ecp->recording_full_grid) {}
}

void Recorder::b_load_intermediate_data_location_slot() {
    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to load intermediate data.");
        return;
    }
    QFileDialog file_dialog{};
    auto dir = QCoreApplication::applicationDirPath() + "/temp";

    auto dir_name = file_dialog.getExistingDirectory(this, tr("Open directory"), dir);
    if (dir_name.toStdString().empty()) {
        return;
    }

    int number_of_files = 0;
    int total_recorded = 0;
    int largest_buffer = 0;

    for (auto & file: std::filesystem::directory_iterator(dir_name.toStdString())) {
        int width;
        int height;
        int pos;
        auto path = file.path().string();
        recd->load_info_buffer_data(path, width, height, pos);
        if (width != edc->simulation_width || height != edc->simulation_height) {
            display_message("Loaded recording has dimensions: " + std::to_string(width) + "x" +
            std::to_string(height) + ". Resize simulation grid to load this recording.");
            return;
        }

        if (largest_buffer < pos) {largest_buffer = pos;}
        number_of_files++;
        total_recorded+=pos;
    }

    recd->buffer_size = largest_buffer;
    recd->buffer_pos = 0;

    recd->second_simulation_grid_buffer.clear();
    recd->second_simulation_grid_buffer.resize(recd->buffer_size, std::vector<BaseGridBlock>(edc->simulation_height*edc->simulation_width));
    recd->path_to_save = dir_name.toStdString();
    recd->saved_buffers = number_of_files;

    recd->recorded_states = total_recorded;
    recording_paused = true;
}

void Recorder::b_compile_intermediate_data_into_video_slot() {
    if (!recording_paused) {
        display_message("Program is still recording. Pause the recording first to compile intermediate data into video.");
        return;
    }

    if (recd->path_to_save.empty()) {
        display_message("No recording is loaded.");
        return;
    }
    ecp->record_full_grid = false;
    while (ecp->recording_full_grid) {}
    engine->pause();

    //Will be loaded from disk
    recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc->simulation_width, edc->simulation_height, recd->second_simulation_grid_buffer);

    std::vector<unsigned char> image_vec(edc->simulation_width*edc->simulation_height*num_pixels_per_block*num_pixels_per_block*4);

    std::string images_path_to_save = recd->path_to_save + "_img";
    std::filesystem::create_directory(images_path_to_save);

    std::filesystem::path p(recd->path_to_save);
    std::string dir_name = p.filename().string();

    auto point = std::chrono::high_resolution_clock::now();
#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_SHOW);
#endif

    std::vector<std::pair<int, std::string>> directories;

    for (auto & file: std::filesystem::directory_iterator(recd->path_to_save)) {
        directories.emplace_back(std::stoi(file.path().filename().string()), file.path().string());
    }

    //file paths do not come out in order, so they need to be sorted first
    std::sort(directories.begin(), directories.end(), [](std::pair<int, std::string> & a, std::pair<int, std::string> & b){
        return a.first < b.first;
    });

    int nums = std::to_string(recd->recorded_states).length();

    int frame_num = 0;
    for (auto & [_, file]: directories) {
        int width;
        int height;
        int len;
        auto path = file;
        recd->load_info_buffer_data(path, width, height, len);
        //Extending buffer if needed
        if (len > recd->buffer_size) {recd->second_simulation_grid_buffer.resize(recd->buffer_size, std::vector<BaseGridBlock>(edc->simulation_height*edc->simulation_width));}
        recd->load_buffer_from_disk(path, width, height, recd->buffer_size, len, recd->second_simulation_grid_buffer);

        for (int i = 0; i < len; i++) {
            frame_num++;

            //Constructing filepath.
            //Because windows has no glob, I need to use %nd thing. To use %nd thing every picture should be
            //n num long. For example if I want to use %4d files need to be 0001.png, 0002.png ...
            //It also needs to be consistent, so I am first creating padding, and add it to the string num.
            std::string padding = std::to_string(frame_num);
            std::string frame_str;
            for (int i = 0; i < nums - frame_str.length(); i++) {frame_str += "0";}
            frame_str += padding;
            std::string image_path;
            image_path.append(images_path_to_save);
            image_path.append("/");
            image_path.append(frame_str);
            image_path.append(".png");
//            image_path = image_path+images_path_to_save+"/"+frame_str+".png";

            if (std::filesystem::exists(image_path)) { continue;}

            create_image(image_vec, recd->second_simulation_grid_buffer[i]);

            QImage image(image_vec.data(),
                         edc->simulation_width*num_pixels_per_block,
                         edc->simulation_height*num_pixels_per_block,
                         QImage::Format_RGB32);

            image.save(QString::fromStdString(image_path), "PNG");
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - point).count() / 1000. > 1./30) {
                clear_console();
                std::cout << "Compiled images " << frame_num <<  "/" << recd->recorded_states << ". Do not turn off program.\n";
            }
        }
    }

    auto program_root = QCoreApplication::applicationDirPath().toStdString();

    std::string ffmpeg_command = ffmpeg_path + " -framerate 60 -start_number 1 -i \"" + images_path_to_save + "/%"+ std::to_string(nums) +"d.png\" -c:v libx264 -pix_fmt yuv420p "+ program_root +"/videos/" + dir_name + ".mp4 -y";

    system(ffmpeg_command.c_str());

    engine->unpause();

#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_SHOW);
#endif
}

void Recorder::b_clear_intermediate_data_slot() {
    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to clear intermediate data.");
        return;
    }
    recd->buffer_pos = 0;
    recd->recorded_states = 0;
    recd->path_to_save = "";
    recording_paused = false;

    recd->second_simulation_grid_buffer.clear();
    recd->second_simulation_grid_buffer.shrink_to_fit();
}

void Recorder::b_delete_all_intermediate_data_from_disk_slot() {
    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to delete intermediate data.");
        return;
    }
    b_clear_intermediate_data_slot();

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
    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to start new recording.");
        return;
    }
    b_clear_intermediate_data_slot();
    auto path = QCoreApplication::applicationDirPath().toStdString();
    recd->path_to_save = new_recording(path);
}