//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

//==================== Line edits ====================

void Recorder::le_number_of_pixels_per_block_slot() {
    le_slot_lower_bound<int>(num_pixels_per_block, num_pixels_per_block, "int", ui.le_number_or_pixels_per_block, 1, "1");
}

void Recorder::le_first_grid_buffer_size_slot() {
    int temp;
    le_slot_lower_bound<int>(temp, temp, "int", ui.le_first_grid_buffer_size, 1, "1");

    if (temp == tbuffer->buffer_pos) { return;}

    if (temp <= tbuffer->buffer_pos) {
        tbuffer->flush_transactions();
    }

    tbuffer->resize_buffer(temp);
    buffer_size = temp;
}

void Recorder::le_log_every_n_tick_slot() {
    int temp = ecp->parse_full_grid_every_n;
    le_slot_lower_bound<int>(temp, temp, "int", ui.le_log_every_n_tick, 1, "1");
    ecp->parse_full_grid_every_n = temp;
}

void Recorder::le_video_fps_slot() {
    le_slot_lower_bound<int>(video_fps, video_fps, "int", ui.le_video_fps, 1, "1");
}

void Recorder::le_zoom_slot() {
    le_slot_lower_bound<float>(zoom, zoom, "float", ui.le_zoom, 0, "0");
}

void Recorder::le_viewpoint_y_slot() {
    le_slot_no_bound(viewpoint_y, viewpoint_y, "float", ui.le_viewpoint_y);
}

void Recorder::le_viewpoint_x_slot() {
    le_slot_no_bound(viewpoint_x, viewpoint_x, "float", ui.le_viewpoint_x);
}

void Recorder::le_image_width_slot() {
    le_slot_lower_bound<int>(image_width, image_width, "int", ui.le_image_width, 1, "1");
}

void Recorder::le_image_height_slot() {
    le_slot_lower_bound<int>(image_height, image_height, "int", ui.le_image_height, 1, "1");
}

void Recorder::le_kernel_size() {
    le_slot_lower_bound<int>(kernel_size, kernel_size, "int", ui.le_kernel_size, 1, "1");
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

    std::vector<unsigned char> raw_image_data;
    int image_width_dim;
    int image_height_dim;

    if (use_viewpoint) {
        raw_image_data.resize(image_width * image_height * 4);
        image_width_dim  = image_width;
        image_height_dim = image_height;
    } else {
        raw_image_data.resize(edc->simulation_width * edc->simulation_height *num_pixels_per_block * num_pixels_per_block * 4);
        image_width_dim  = edc->simulation_width*num_pixels_per_block;
        image_height_dim = edc->simulation_height*num_pixels_per_block;
    }

    engine->parse_full_simulation_grid();

    create_image(raw_image_data, edc->simple_state_grid, edc->simulation_width, edc->simulation_height,
                 num_pixels_per_block, false, use_viewpoint, false, kernel_size);

    QImage image(raw_image_data.data(),
                 image_width_dim,
                 image_height_dim,
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
            auto type = edc->st_grid.get_type(x, y);
            if (type == BlockTypes::WallBlock) {
                tbuffer->record_user_wall_change(x, y, true);
            }
        }
    }

    recording_paused = false;
    edc->record_data = true;
}

void Recorder::b_stop_recording_slot() {
    edc->record_data = false;
    ecp->lock_resizing = false;
    recording_paused = false;

    tbuffer->finish_recording();

    clear_data();
}

void Recorder::b_pause_recording_slot() {
    if (!edc->record_data) {
        display_message("Start the recording first.");
        return;
    }

    ecp->tb_paused = true;
    engine->pause();
    engine->wait_for_engine_to_pause();
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

    ui.le_first_grid_buffer_size->setText(QString::fromStdString(std::to_string(tbuffer->buffer_size)));

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

    //TODO
    if (compiling_recording) {
        display_message("Recording is being compiled.");
        return;
    }

    compiling_recording = true;
    edc->record_data = false;

    //Will be loaded from disk
    tbuffer->flush_transactions();

    start_normal_thread();

    b_stop_recording_slot();
    tbuffer->finish_recording();

#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif
}

void Recorder::start_normal_thread() {
    std::thread thr2([this](std::string path_to_save,
                            int simulation_width,
                            int simulation_height,
                            int num_pixels_per_block,
                            int recorded_states,
                            int buffer_size,
                            int video_fps,
                            int image_width,
                            int image_height,
                            bool use_cuda,
                            bool cuda_is_available,
                            bool use_viewpoint) {
        int modifier = 4;
        if (use_cuda) {modifier = 3;}

         std::vector<unsigned char> image_vec;

        if (use_viewpoint) {
            image_vec = std::vector<unsigned char>(
                    image_width * image_height * modifier);
        } else {
            image_vec = std::vector<unsigned char>(
                    simulation_width * simulation_height * num_pixels_per_block * num_pixels_per_block * modifier);
        }

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
            int dim_width;
            int dim_height;
            if (use_viewpoint) {
                dim_width = image_width;
                dim_height = image_height;
            } else {
                dim_width = simulation_width * num_pixels_per_block;
                dim_height = simulation_height * num_pixels_per_block;
            }

            writer.start_writing(movie_name, dim_width, dim_height, video_fps);
//        }

        int processed_transactions = 0;
        int frame_num = 0;
        int last_frame_num = 0;

#ifdef __CUDA_USED__
        RecordingReconstructorCUDA cuda_reconstructor{};
        if (use_cuda_reconstructor && use_cuda && cuda_is_available) {
            int start_x;
            int end_x;
            int start_y;
            int end_y;
            std::vector<int> lin_width;
            std::vector<int> lin_height;
            std::vector<int> truncated_lin_width;
            std::vector<int> truncated_lin_height;

            std::vector<Vector2<int>> width_img_boundaries;
            std::vector<Vector2<int>> height_img_boundaries;

            if (!use_viewpoint) {
                prepare_full_view(simulation_width, simulation_height, num_pixels_per_block, image_height, start_x, end_x,
                                  start_y,
                                  end_y, truncated_lin_width, truncated_lin_height, image_width, lin_width, lin_height);
            } else {
                prepare_relative_view(lin_height, truncated_lin_width,
                                      truncated_lin_height, image_width, image_height, start_x, end_x, start_y, end_y,
                                      lin_width);
            }

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

            cuda_reconstructor.start_reconstruction(simulation_width, simulation_height);
            cuda_reconstructor.prepare_image_creation(image_width, image_height,
                                                      lin_width, lin_height,
                                                      width_img_boundaries, height_img_boundaries,
                                                      *textures, cc);
        } else {
#endif
            reconstructor.start_reconstruction(simulation_width, simulation_height);
#ifdef __CUDA_USED__
        }
#endif
        for (auto &[_, file]: directories) {
            int width;
            int height;
            int len;
            auto path = file;
            local_tbuffer.load_buffer_metadata(path, width, height, len);
            local_tbuffer.load_buffer(path);

            for (int i = 0; i < len; i++) {
                processed_transactions++;
#ifdef __CUDA_USED__
                if (use_cuda_reconstructor && use_cuda && cuda_is_available) {
                    cuda_reconstructor.apply_transaction(local_tbuffer.transactions[i]);
                } else {
#endif
                    reconstructor.apply_transaction(local_tbuffer.transactions[i]);
#ifdef __CUDA_USED__
                }
#endif

                if (processed_transactions <= loaded_frames) { continue;}
                if (processed_transactions % parse_every != 0) { continue;}

                frame_num++;
#ifdef __CUDA_USED__
                if (use_cuda_reconstructor && use_cuda && cuda_is_available) {
                    cuda_reconstructor.make_image(image_vec);
                } else {
#endif
                    create_image(image_vec, reconstructor.get_state(), simulation_width, simulation_height,
                                 num_pixels_per_block, use_cuda, use_viewpoint, use_cuda, 1);
#ifdef __CUDA_USED__
                }
#endif

                if (use_cuda) {
                    writer.addYUVFrame(image_vec.data());
                } else {
                    writer.addFrame(image_vec.data());
                }

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
                    ". Compiled frames " << frame_num << "/" << recorded_states/parse_every << ". Frames in second "
                    << frames_in_second << ". Expected time until completion: " << time
                              << ". Do not turn off program.\n";
                }
            }
        }
#ifdef __CUDA_USED__
        if (use_cuda_reconstructor && use_cuda && cuda_is_available) {
            cuda_reconstructor.finish_reconstruction();
            cuda_reconstructor.finish_image_creation();
        } else {
#endif
            reconstructor.finish_reconstruction();
            compiling_recording = false;
#ifdef __CUDA_USED__
        }
#endif
    }, tbuffer->path_to_save, edc->simulation_width, edc->simulation_height,
                     num_pixels_per_block, tbuffer->recorded_transactions, tbuffer->buffer_size, video_fps,
                     image_width, image_height, use_cuda, cuda_is_available, use_viewpoint);

    thr2.detach();
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

void Recorder::b_set_from_camera_slot() {
    viewpoint_x = *main_viewpoint_x;
    viewpoint_y = *main_viewpoint_y;
    zoom = *main_zoom;
    image_width  = parent_ui->simulation_graphicsView->viewport()->width();
    image_height = parent_ui->simulation_graphicsView->viewport()->height();

    ui.le_viewpoint_x->setText(QString::fromStdString(to_str(viewpoint_x, 5)));
    ui.le_viewpoint_y->setText(QString::fromStdString(to_str(viewpoint_y, 5)));
    ui.le_zoom->setText(QString::fromStdString(to_str(zoom, 5)));

    ui.le_image_width ->setText(QString::fromStdString(std::to_string(image_width)));
    ui.le_image_height->setText(QString::fromStdString(std::to_string(image_height)));
}

void Recorder::cb_use_relative_viewpoint_slot(bool state) {use_viewpoint = state; b_set_from_camera_slot();}

void Recorder::cb_use_cuda_slot(bool state) {
    if (!state) {
        use_cuda = false;
#ifdef __CUDA_USED__
        cuda_image_creator.free();
#endif
        return;}

    if (!cuda_is_available) {
        ui.cb_use_cuda->setChecked(false);
        use_cuda = false;
        display_message("Warning, CUDA is not available on this device.");
        return;
    }
    use_cuda = true;
#ifdef __CUDA_USED__
    cuda_image_creator.copy_textures(*textures);
#endif
}

void Recorder::cb_use_cuda_reconstructor_slot(bool state) {
    use_cuda_reconstructor = state;
}