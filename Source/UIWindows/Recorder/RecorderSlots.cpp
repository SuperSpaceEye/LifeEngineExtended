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

    if (temp == tbuffer->buffer_pos) { return;}

    if (temp <= tbuffer->buffer_pos) {
        tbuffer->flush_transactions();
    }

    tbuffer->resize_buffer(temp);
    buffer_size = temp;
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

    create_image(raw_image_data, edc->simple_state_grid, edc->simulation_width, edc->simulation_height, num_pixels_per_block);

    QImage image(raw_image_data.data(),
                 edc->simulation_width*num_pixels_per_block,
                 edc->simulation_height*num_pixels_per_block,
                 QImage::Format_RGB32);

    image.save(QString::fromStdString(full_path), "PNG");

    engine->unpause();
}

void Recorder::b_start_recording_slot() {
    //if already recording
    if (edc->record_data) { return;}
    if (tbuffer->path_to_save.empty()) {
        display_message("Make new recording to start.");
        return;
    }
    ecp->lock_resizing = true;

    if (!recording_paused) {
        tbuffer->start_recording(tbuffer->path_to_save, edc->simulation_width, edc->simulation_height, buffer_size);
    }

    for (auto & o: edc->stc.organisms) {
        if (!o.is_dead) {
            tbuffer->record_new_organism(o);
        }
    }

    for (int x = 0; x < edc->simulation_width; x++) {
        for (int y = 0; y < edc->simulation_height; y++) {
            auto & block = edc->CPU_simulation_grid[x][y];
            if (block.type == BlockTypes::FoodBlock) {
                tbuffer->record_food_change(x, y, true);
            } else if (block.type == BlockTypes::WallBlock) {
                tbuffer->record_wall_changes(x, y, true);
            }
        }
    }

    recording_paused = false;
    edc->record_data = true;
}

void Recorder::b_stop_recording_slot() {
    edc->record_data = false;
    ecp->lock_resizing = false;

    tbuffer->finish_recording();

    clear_data();
}

void Recorder::b_pause_recording_slot() {
    if (!edc->record_data) {
        display_message("Start the recording first.");
        return;
    }

    ecp->pause_button_pause = true;
    ecp->tb_paused = true;
    engine->pause();
    engine->wait_for_engine_to_pause_force();
    engine->unpause();
    parent_ui->tb_pause->setChecked(true);

    edc->record_data = false;
    recording_paused = true;
    tbuffer->record_transaction();

    tbuffer->record_reset();
}

void Recorder::b_load_intermediate_data_location_slot() {
    if (edc->record_data && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to load intermediate data.");
        return;
    }
    QFileDialog file_dialog{};
    auto dir = QCoreApplication::applicationDirPath() + "/temp";

    auto dir_name = file_dialog.getExistingDirectory(this, tr("Open directory"), dir);
    if (dir_name.toStdString().empty()) {
        return;
    }

    tbuffer->finish_recording();

    int number_of_files = 0;
    int total_recorded = 0;
    int largest_buffer = 0;

    for (auto & file: std::filesystem::directory_iterator(dir_name.toStdString())) {
        int width;
        int height;
        int pos;
        auto path = file.path().string();
        tbuffer->load_buffer_metadata(path, width, height, pos);
        if (width != edc->simulation_width || height != edc->simulation_height) {
            display_message("Loaded recording has dimensions: " + std::to_string(width) + "x" +
            std::to_string(height) + ". Resize simulation grid to load this recording.");
            return;
        }

        if (largest_buffer < pos) {largest_buffer = pos;}
        number_of_files++;
        total_recorded+=pos;
    }

    tbuffer->resize_buffer(largest_buffer);
    tbuffer->start_recording(dir_name.toStdString(), edc->simulation_width, edc->simulation_height, tbuffer->buffer_size);
    tbuffer->recorded_transactions = total_recorded;
    tbuffer->saved_buffers = number_of_files;

    _ui.le_first_grid_buffer_size->setText(QString::fromStdString(std::to_string(tbuffer->buffer_size)));

    recording_paused = true;
}

void Recorder::b_compile_intermediate_data_into_video_slot() {
    if (!display_dialog_message("Compile recording into video?", false)) {return;}

    if (!recording_paused) {
        display_message("Program is still recording. Pause the recording first to compile recording into video.");
        return;
    }

    if (tbuffer->path_to_save.empty()) {
        display_message("No recording is loaded.");
        return;
    }
    edc->record_data = false;

//    lock_recording = true;

    //Will be loaded from disk
    tbuffer->flush_transactions();

    std::thread thr2([this](std::string path_to_save,
                            int simulation_width,
                            int simulation_height,
                            int num_pixels_per_block,
                            int recorded_states,
                            int buffer_size,
                            int video_fps) {
        std::vector<unsigned char> image_vec(
             simulation_width * simulation_height * num_pixels_per_block * num_pixels_per_block * 4);

        TransactionBuffer local_tbuffer;
        int parse_every = ecp->parse_full_grid_every_n;

        auto point = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<int, std::string>> directories;

        for (auto &file: std::filesystem::directory_iterator(path_to_save)) {
         directories.emplace_back(std::stoi(file.path().filename().string()), file.path().string());
        }

        //file paths do not come out in order, so they need to be sorted first
        std::sort(directories.begin(), directories.end(),
               [](std::pair<int, std::string> &a, std::pair<int, std::string> &b) {
                   return a.first < b.first;
               });


        std::filesystem::path p(path_to_save);
        std::string dir_name = p.filename().string();
        auto program_root = QCoreApplication::applicationDirPath().toStdString();
        std::string movie_name = program_root + "/videos/" + dir_name;

        int loaded_frames = 0;

        MovieWriter writer;

        //TODO
//        if (std::filesystem::exists(movie_name+".mp4")) {
//            std::filesystem::rename(movie_name+".mp4", movie_name+"_temp.mp4");
//            writer.start_writing(movie_name, simulation_width * num_pixels_per_block,
//                                 simulation_height * num_pixels_per_block, video_fps);
//            {
//                MovieReader reader(movie_name + "_temp", simulation_width * num_pixels_per_block,
//                                   simulation_height * num_pixels_per_block);
//
//                while (reader.getFrame(image_vec)) {
//                    writer.addFrame(&image_vec[0]);
//                    loaded_frames++;
//
//                    auto point2 = std::chrono::high_resolution_clock::now();
//                    if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - point).count() / 1000. > 1. / 5) {
//                        point = std::chrono::high_resolution_clock::now();
//                        clear_console();
//                        std::cout << "Loading frames " << loaded_frames << ". Do not turn off program.\n";
//                    }
//                }
//
//                reader.stop_reading();
//            }
//
//            std::filesystem::remove(movie_name+"_temp.mp4");
//        } else {
            writer.start_writing(movie_name, simulation_width * num_pixels_per_block,
                                 simulation_height * num_pixels_per_block, video_fps);
//        }

        int processed_transactions = 0;
        int frame_num = 0;
        int last_frame_num = 0;

        reconstructor.start_reconstruction(simulation_width, simulation_height);
        for (auto &[_, file]: directories) {
            int width;
            int height;
            int len;
            auto path = file;
            local_tbuffer.load_buffer_metadata(path, width, height, len);
            local_tbuffer.load_buffer(path);

            for (int i = 0; i < len; i++) {
                processed_transactions++;

                reconstructor.apply_transaction(local_tbuffer.transactions[i]);

                if (processed_transactions <= loaded_frames) { continue;}
                if (processed_transactions % parse_every != 0) { continue;}

                frame_num++;

                Recorder::create_image(image_vec, reconstructor.get_state(), simulation_width, simulation_height, num_pixels_per_block);

                writer.addFrame(&image_vec[0]);

                auto point2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - point).count() / 1000. > 1. / 5) {
                    int frame_diff = frame_num - last_frame_num;
                    if (frame_diff <= 0) {frame_diff = 1;}
                    last_frame_num = frame_num;

                    auto scale = std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point).count() / 1000.;
                    int frames_in_second = frame_diff / scale;
                    if (frames_in_second <= 0) {frames_in_second = 1;}
                    auto time = convert_seconds((recorded_states / parse_every - frame_num) / frames_in_second);

                    point = std::chrono::high_resolution_clock::now();
                    clear_console();
                    std::cout << "Processed transactions " << processed_transactions << "/" << recorded_states <<
                    ". Compiled frames " << frame_num << "/" << recorded_states/parse_every << ". Expected time until completion: " << time
                              << ". Do not turn off program.\n";
                }
            }
        }
        reconstructor.finish_reconstruction();
    }, tbuffer->path_to_save, edc->simulation_width, edc->simulation_height,
       num_pixels_per_block, tbuffer->recorded_transactions, tbuffer->buffer_size, video_fps);

    thr2.detach();

    b_stop_recording_slot();
    tbuffer->finish_recording();

#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif
}

void Recorder::b_clear_intermediate_data_slot() {
    if (!display_dialog_message("Clear intermediate data?", false)) {return;}

    if (edc->record_data && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to clear saved recording.");
        return;
    }
    clear_data();
}

void Recorder::b_delete_all_intermediate_data_from_disk_slot() {
    if (!display_dialog_message("Delete the recording from disk?", false)) {return;}
    if (!display_dialog_message("Are you sure?", false)) {return;}

    if (edc->record_data && !recording_paused) {
        display_message("Program is still recording. Stop the recording first to delete the recording.");
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
    if (edc->record_data || recording_paused) {
        display_message("Program is still recording. Stop the recording first to start new recording.");
        return;
    }

    clear_data();
    auto path = QCoreApplication::applicationDirPath().toStdString();
    tbuffer->path_to_save = new_recording(path);
}