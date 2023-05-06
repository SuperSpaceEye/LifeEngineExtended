//
// Created by spaceeye on 29.03.23.
//

#include "MainWindow.h"

void MainWindow::create_image_creation_thread() {
    image_creation_thread = std::thread{[&](){
        auto point1 = std::chrono::high_resolution_clock::now();
        auto point2 = point1;
        while (ecp.make_images) {
            point1 = std::chrono::high_resolution_clock::now();
            if (!pause_grid_parsing || !really_stop_render) {
                if (have_read_buffer) {
                    have_read_buffer = false;
                    create_image();
                    image_frames++;
                }
            }
            point2 = std::chrono::high_resolution_clock::now();
            std::this_thread::sleep_for(std::chrono::microseconds(
                    std::max<int64_t>(
                            int(image_creation_interval * 1000000) -
                            std::chrono::duration_cast<std::chrono::microseconds>(point2 - point1).count()
            , 0)));
        }
    }};

    image_creation_thread.detach();
}

void MainWindow::create_image() {
    while (do_not_parse_image_data_mt) {}
    do_not_parse_image_data_ct.store(true);

    std::vector<int> lin_width;
    std::vector<int> lin_height;
    std::vector<int> truncated_lin_width;
    std::vector<int> truncated_lin_height;

    pre_parse_simulation_grid_stage(lin_width, lin_height, truncated_lin_width, truncated_lin_height);

    parse_simulation_grid_stage(truncated_lin_width, truncated_lin_height);

    int new_buffer = !bool(ready_buffer);

#ifdef __CUDA_USED__
    void * cuda_creator_ptr = &cuda_creator;
#else
    void * cuda_creator_ptr = nullptr;
#endif

    ImageCreation::create_image(lin_width, lin_height, edc.simulation_width, edc.simulation_height, cc, textures,
                                image_width, image_height, image_vectors[new_buffer], edc.simple_state_grid,
                                use_cuda, cuda_is_available_var, cuda_creator_ptr, truncated_lin_width,
                                truncated_lin_height, false, 1);

    ready_buffer = new_buffer;
    do_not_parse_image_data_ct.store(false);
}

void MainWindow::parse_simulation_grid_stage(const std::vector<int> &truncated_lin_width,
                                             const std::vector<int> &truncated_lin_height) {
    if (!pause_grid_parsing && !ecp.engine_global_pause) {
        parse_simulation_grid(truncated_lin_width, truncated_lin_height);
    }
}

void MainWindow::pre_parse_simulation_grid_stage(std::vector<int> &lin_width, std::vector<int> &lin_height,
                                                 std::vector<int> &truncated_lin_width,
                                                 std::vector<int> &truncated_lin_height) {
    image_width  = ui.simulation_graphicsView->viewport()->width();
    image_height = ui.simulation_graphicsView->viewport()->height();
    resize_image(image_width, image_height);
    int scaled_width  = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and y coordinates on simulation grid
    auto start_x = int(center_x - (scaled_width / 2));
    auto end_x   = int(center_x + (scaled_width / 2));

    auto start_y = int(center_y - (scaled_height / 2));
    auto end_y   = int(center_y + (scaled_height / 2));
    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    truncated_lin_width .reserve(abs(lin_width [lin_width.size() - 1]) + 1);
    truncated_lin_height.reserve(abs(lin_height[lin_height.size() - 1]) + 1);

    ImageCreation::calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);
}

void MainWindow::parse_simulation_grid(const std::vector<int> &lin_width, const std::vector<int> &lin_height) {
    for (int x: lin_width) {
        if (x < 0 || x >= edc.simulation_width) { continue; }
        for (int y: lin_height) {
            if (y < 0 || y >= edc.simulation_height) { continue; }
            auto type = edc.st_grid.get_type(x, y);
            auto & simple_block = edc.simple_state_grid[x + y * edc.simulation_width];
            simple_block.type = type;
            simple_block.rotation = edc.st_grid.get_rotation(x, y);

            if (type == BlockTypes::EmptyBlock && edc.st_grid.get_food_num(x, y) >= sp.food_threshold) {
                simple_block.type = BlockTypes::FoodBlock;}
        }
    }
}