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
#include "textures.h"
#include "../Containers/CPU/ColorContainer.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Stuff/Vector2.h"

struct Differences {
    uint32_t x;
    uint32_t y;
    BlockTypes type;
    Rotation rotation;
};

struct CudaTextureHolder {
    int width;
    int height;
    color* texture;
};

class CUDAImageCreator {
    int * d_lin_width = nullptr;
    int * d_lin_height = nullptr;

    Vector2<int> *d_width_img_boundaries = nullptr;
    Vector2<int> *d_height_img_boundaries = nullptr;
    unsigned char * d_image_vector = nullptr;
    BaseGridBlock * d_second_simulation_grid = nullptr;
    Differences * d_differences = nullptr;
    CudaTextureHolder * d_textures = nullptr;

    std::vector<color*> d_textures_pointers;

    std::vector<BaseGridBlock> device_state_grid{};

    int last_image_width = 0;
    int last_image_height = 0;
    int last_lin_width = 0;
    int last_lin_height = 0;

    int last_width_img_boundaries = 0;
    int last_height_img_boundaries = 0;
    int last_simulation_width = 0;
    int last_simulation_height = 0;
    int last_differences = 0;

    volatile bool creating_image = false;
    volatile bool do_not_create_image = false;

    void image_dimensions_changed(int image_width, int image_height);

    void simulation_dimensions_changed(int simulation_width, int simulation_height);

    void img_boundaries_changed(int width_img_boundaries_size, int height_img_boundaries_size);

    void lin_size_changed(int lin_width_size, int lin_height_size);

    void differences_changed(int differences);

    void free_textures();

    void check_if_changed(int image_width, int image_height, int simulation_width, int simulation_height,
                          int width_img_boundaries_size, int height_img_boundaries_size,
                          int lin_width_size, int lin_height_size, int differences);

    void copy_to_device(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                        const std::vector<Vector2<int>> &width_img_boundaries,
                        const std::vector<Vector2<int>> &height_img_boundaries,
                        const std::vector<Differences> &host_differences);

    void copy_result_image(std::vector<unsigned char> &image_vector, int image_width, int image_height);

public:
    CUDAImageCreator()=default;

    void free();

    void compile_differences(const std::vector<int> &truncated_lin_width, const std::vector<int> &truncated_lin_height,
                             std::vector<Differences> &host_differences, int simulation_width,
                             int simulation_height, const std::vector<BaseGridBlock> &simple_state_grid);

    void cuda_create_image(int image_width, int image_height, const std::vector<int> &lin_width,
                           const std::vector<int> &lin_height, std::vector<unsigned char> &image_vector,
                           const ColorContainer &color_container, int block_size, int simulation_width,
                           int simulation_height, std::vector<Differences> &differences);

    void load_symbols(const ColorContainer *colorContainer);

    void copy_textures(TexturesContainer & container);
};

#endif //THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
