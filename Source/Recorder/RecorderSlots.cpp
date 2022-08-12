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
        recd->buffer_pos = 0;
    }
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

void Recorder::le_video_fps_slot() {
    le_slot_lower_bound<int>(video_fps, video_fps, "int", _ui.le_video_fps, 1, "1");
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

    create_image(raw_image_data, edc->second_simulation_grid, edc->simulation_width, edc->simulation_height, num_pixels_per_block);

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
    ecp->lock_resizing = true;

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
    ecp->lock_resizing = false;

    if (recd->path_to_save.empty()) { return;}
    recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc->simulation_width, edc->simulation_height, recd->second_simulation_grid_buffer);

    clear_data();
}

void Recorder::b_pause_recording_slot() {
    if (!ecp->record_full_grid) {
        display_message("Start the recording first.");
        return;
    }

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
    _ui.le_first_grid_buffer_size->setText(QString::fromStdString(std::to_string(recd->buffer_size)));

    recd->second_simulation_grid_buffer.clear();
    recd->second_simulation_grid_buffer.resize(recd->buffer_size, std::vector<BaseGridBlock>(edc->simulation_height*edc->simulation_width));
    recd->path_to_save = dir_name.toStdString();
    recd->saved_buffers = number_of_files;

    recd->recorded_states = total_recorded;
    recording_paused = true;
}

void Recorder::b_compile_intermediate_data_into_video_slot() {
    if (!display_dialog_message("Compile recording into video?", false)) {return;}

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

//    lock_recording = true;

    //Will be loaded from disk
    recd->save_buffer_to_disk(recd->path_to_save, recd->buffer_pos, recd->saved_buffers, edc->simulation_width, edc->simulation_height, recd->second_simulation_grid_buffer);

    std::thread thr([this](
            std::string path_to_save,
            int simulation_width,
            int simulation_height,
            int num_pixels_per_block,
            int recorded_states,
            int buffer_size,
            int * video_fps
            ) {
        std::vector<unsigned char> image_vec(
                simulation_width * simulation_height * num_pixels_per_block * num_pixels_per_block * 4);

        std::vector<std::vector<BaseGridBlock>> local_buffer;

        std::string images_path_to_save = path_to_save + "_img";
        std::filesystem::create_directory(images_path_to_save);

        std::filesystem::path p(path_to_save);
        std::string dir_name = p.filename().string();

        auto point = std::chrono::high_resolution_clock::now();
#if defined(__WIN32)
        ShowWindow(GetConsoleWindow(), SW_SHOW);
        auto console = GetConsoleWindow();
        RECT r;
        GetWindowRect(console, &r);

        MoveWindow(console, r.left, r.top, 200, 100, TRUE);

        CONSOLE_FONT_INFOEX cfi;
        cfi.cbSize = sizeof(cfi);
        cfi.nFont = 0;
        cfi.dwFontSize.X = 0;                   // Width of each character in the font
        cfi.dwFontSize.Y = 24;                  // Height
        cfi.FontFamily = FF_DONTCARE;
        cfi.FontWeight = FW_NORMAL;
        std::wcscpy(cfi.FaceName, L"Consolas"); // Choose your font
        SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &cfi);
#endif

        std::vector<std::pair<int, std::string>> directories;

        for (auto &file: std::filesystem::directory_iterator(path_to_save)) {
            directories.emplace_back(std::stoi(file.path().filename().string()), file.path().string());
        }

        //file paths do not come out in order, so they need to be sorted first
        std::sort(directories.begin(), directories.end(),
                  [](std::pair<int, std::string> &a, std::pair<int, std::string> &b) {
                      return a.first < b.first;
                  });

        int nums = std::to_string(recorded_states).length();

        int frame_num = 0;
        int last_frame_num = 0;
        for (auto &[_, file]: directories) {
            int width;
            int height;
            int len;
            auto path = file;
            RecordingData::load_info_buffer_data(path, width, height, len);
            //Extending buffer if needed
            if (len > local_buffer.size()) { local_buffer.resize(len,
                                                                 std::vector<BaseGridBlock>(
                                                                         simulation_height *
                                                                         simulation_width));
                buffer_size = len;
            }
            RecordingData::load_buffer_from_disk(path, width, height, buffer_size, len, local_buffer);



            for (int i = 0; i < len; i++) {
                frame_num++;

                //Constructing filepath.
                //Because windows has no glob, I need to use %nd thing. To use %nd thing every picture should be
                //n num long. For example if I want to use %4d files need to be 0001.png, 0002.png ...
                //It also needs to be consistent, so I am first creating padding, and add it to the string num.
                std::string padding = std::to_string(frame_num);
                std::string frame_str;
                for (int i = 0; i < nums - padding.length(); i++) { frame_str += "0"; }
                frame_str += padding;
                std::string image_path;
                image_path.append(images_path_to_save);
                image_path.append("/");
                image_path.append(frame_str);
                image_path.append(".png");
//            image_path = image_path+images_path_to_save+"/"+frame_str+".png";

                if (std::filesystem::exists(image_path)) { continue; }

                Recorder::create_image(image_vec, local_buffer[i], simulation_width, simulation_height, num_pixels_per_block);

                QImage image(image_vec.data(),
                             simulation_width * num_pixels_per_block,
                             simulation_height * num_pixels_per_block,
                             QImage::Format_RGB32);

                image.save(QString::fromStdString(image_path), "PNG");
                auto point2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - point).count() / 1000. > 1. / 5) {
                    int frame_diff = frame_num - last_frame_num;
                    if (frame_diff <= 0) {frame_diff = 1;}
                    last_frame_num = frame_num;

                    auto scale = std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point).count() / 1000.;
                    int frames_in_second = frame_diff / scale;
                    if (frames_in_second <= 0) {frames_in_second = 1;}
                    auto time = convert_seconds((recorded_states - frame_num)/frames_in_second);

                    point = std::chrono::high_resolution_clock::now();
                    clear_console();
                    std::cout << "Compiled images " << frame_num << "/" << recorded_states << ". Expected time until completion: " << time
                              << ". Do not turn off program.\n";
                }
            }
        }

        clear_console();
        std::cout << "Video is being created. Do not turn off the program.\n";

        auto program_root = QCoreApplication::applicationDirPath().toStdString();

        std::string ffmpeg_command =
                ffmpeg_path + " -framerate " + std::to_string(*video_fps) + " -start_number 1 -i \"" +
                images_path_to_save + "/%" + std::to_string(nums) + "d.png\" -c:v libx264 -pix_fmt yuv420p " +
                program_root + "/videos/" + dir_name + ".mp4 -y";

//    #ifdef __WIN32
//    _popen(ffmpeg_command.c_str(), "rt");
//    #elif defined (__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
//    system(ffmpeg_command.c_str());
//    #endif

        auto a = system(ffmpeg_command.c_str());

//        lock_recording = false;
    },
                    recd->path_to_save, edc->simulation_width, edc->simulation_height,
                    num_pixels_per_block, recd->recorded_states, recd->buffer_size, &video_fps);
    thr.detach();
    b_stop_recording_slot();

#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif
}

void Recorder::b_clear_intermediate_data_slot() {
    if (!display_dialog_message("Clear intermediate data?", false)) {return;}

    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to clear intermediate data.");
        return;
    }
    clear_data();
}

void Recorder::b_delete_all_intermediate_data_from_disk_slot() {
    if (!display_dialog_message("Delete intermediate data from disk?", false)) {return;}
    if (!display_dialog_message("Are you sure? It will delete all recording and partially compiled videos.", false)) {return;}

    if (ecp->record_full_grid && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to delete intermediate data.");
        return;
    }
    clear_data();

    auto path = QCoreApplication::applicationDirPath().toStdString() + "/temp";

    std::thread thr([path]() {
        std::vector<std::string> subdirectories;
        for (const auto & entry: std::filesystem::directory_iterator(path)) {
            subdirectories.emplace_back(entry.path().string());
        }

        for (auto &dir: subdirectories) {
            std::filesystem::remove_all(dir);
        }
    });
    thr.detach();
}

void Recorder::b_new_recording_slot() {
    if (ecp->record_full_grid || recording_paused) {
        display_message("Program is still recording. Stop the recording first to start new recording.");
        return;
    }

    clear_data();
    auto path = QCoreApplication::applicationDirPath().toStdString();
    recd->path_to_save = new_recording(path);
}