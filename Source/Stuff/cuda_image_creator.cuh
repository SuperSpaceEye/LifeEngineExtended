//
// Created by spaceeye on 05.06.22.
//

#ifndef THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
#define THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH

#if __WIN32
    #pragma once

    #ifdef CUDA_IMAGE_CREATOR_EXPORTS
        #define CUDA_IMAGE_CREATOR_API __declspec(dllexport)
    #else
        #define CUDA_IMAGE_CREATOR_API __declspec(dllimport)
    #endif
#else
    #define CUDA_IMAGE_CREATOR_API
#endif

#include <vector>
#include "pix_pos.h"
#include "textures.h"
#include "../Containers/CPU/ColorContainer.h"
#include "../Containers/CPU/EngineDataContainer.h"

struct Differences {
    uint16_t x;
    uint16_t y;
    BlockTypes type;
    Rotation rotation;
};

class CUDAImageCreator {
    int * d_lin_width = nullptr;
    int * d_lin_height = nullptr;

    pix_pos *d_width_img_boundaries = nullptr;
    pix_pos *d_height_img_boundaries = nullptr;
    unsigned char * d_image_vector = nullptr;
    BaseGridBlock * d_second_simulation_grid = nullptr;
    Differences * d_differences = nullptr;

    int last_image_width = 0;
    int last_image_height = 0;
    int last_lin_width = 0;
    int last_lin_height = 0;

    int last_width_img_boundaries = 0;
    int last_height_img_boundaries = 0;
    int last_simulation_width = 0;
    int last_simulation_height = 0;
    int last_differences = 0;

    void image_dimensions_changed(int image_width, int image_height);

    void simulation_dimensions_changed(int simulation_width, int simulation_height);

    void img_boundaries_changed(int width_img_boundaries_size, int height_img_boundaries_size);

    void lin_size_changed(int lin_width_size, int lin_height_size);

    void differences_changed(int differences);

    void check_if_changed(int image_width, int image_height, int simulation_width, int simulation_height,
                          int width_img_boundaries_size, int height_img_boundaries_size,
                          int lin_width_size, int lin_height_size, int differences);

    void copy_to_device(std::vector<int> &lin_width, std::vector<int> &lin_height,
                        std::vector<pix_pos> &width_img_boundaries, std::vector<pix_pos> &height_img_boundaries,
                        std::vector<int> & truncated_lin_width,
                        std::vector<int> & truncated_lin_height,
                        std::vector<Differences> &host_differences);

    void compile_differences(std::vector<int> &truncated_lin_width, std::vector<int> &truncated_lin_height,
                             std::vector<Differences> &host_differences, EngineDataContainer *dc);

    void copy_result_image(std::vector<unsigned char> &image_vector, int image_width, int image_height);

public:
    CUDAImageCreator()=default;

    void free();

    void cuda_create_image(int image_width, int image_height, std::vector<int> &lin_width, std::vector<int> &lin_height,
                           std::vector<unsigned char> &image_vector, ColorContainer &color_container,
                           EngineDataContainer &dc, int block_size, std::vector<int> &truncated_lin_width,
                           std::vector<int> &truncated_lin_height);

    void load_symbols(ColorContainer *colorContainer);
};

#endif //THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
