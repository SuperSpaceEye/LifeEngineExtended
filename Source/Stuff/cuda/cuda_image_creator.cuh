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
#include "Stuff/ImageStuff/textures.h"
#include "Containers/ColorContainer.h"
#include "Containers/EngineDataContainer.h"
#include "Stuff/structs/Vector2.h"

struct Differences {
    uint32_t x;
    uint32_t y;
    BlockTypes type;
    Rotation rotation;
};

struct CudaTextureHolder {
    int width;
    int height;
    Textures::color * texture;
};

class CUDAImageCreator {
public:
    int * d_lin_width = nullptr;
    int * d_lin_height = nullptr;

    Vector2<int> *d_width_img_boundaries = nullptr;
    Vector2<int> *d_height_img_boundaries = nullptr;
    unsigned char * d_image_vector = nullptr;
    BaseGridBlock * d_second_simulation_grid = nullptr;
    Differences * d_differences = nullptr;
    CudaTextureHolder * d_textures = nullptr;

    std::vector<Textures::color*> d_textures_pointers;

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

    void image_dimensions_changed(int image_width, int image_height, bool yuv_format);

    void simulation_dimensions_changed(int simulation_width, int simulation_height);

    void img_boundaries_changed(int width_img_boundaries_size, int height_img_boundaries_size);

    void lin_size_changed(int lin_width_size, int lin_height_size);

    void differences_changed(int differences);

    void free_textures();

    void check_if_changed(int image_width, int image_height, int simulation_width, int simulation_height,
                          int width_img_boundaries_size, int height_img_boundaries_size,
                          int lin_width_size, int lin_height_size, int differences, bool yuv_format);

    void copy_to_device(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                        const std::vector<Vector2<int>> &width_img_boundaries,
                        const std::vector<Vector2<int>> &height_img_boundaries,
                        const std::vector<Differences> &host_differences);

    static void copy_result_image(std::vector<unsigned char> &image_vector, int image_width, int image_height,
                                  unsigned char *d_image_vector, bool yuv_format);

public:
    CUDAImageCreator()=default;

    void free();

    void compile_differences(const std::vector<int> &truncated_lin_width, const std::vector<int> &truncated_lin_height,
                             std::vector<Differences> &host_differences, int simulation_width,
                             int simulation_height, const std::vector<BaseGridBlock> &simple_state_grid);

    void cuda_create_image(int image_width, int image_height, const std::vector<int> &lin_width,
                           const std::vector<int> &lin_height, std::vector<unsigned char> &image_vector,
                           const ColorContainer &color_container, int block_size, int simulation_width,
                           int simulation_height, std::vector<Differences> &differences, bool yuv_format);

    void load_symbols(const ColorContainer *colorContainer);

    void copy_textures(Textures::TexturesContainer & container);

    static void
    cuda_call_create_image(int image_width, int image_height, std::vector<unsigned char> &image_vector,
                           int block_size, int simulation_width, int simulation_height, int *d_lin_width,
                           int *d_lin_height, Vector2<int> *d_width_img_boundaries,
                           Vector2<int> *d_height_img_boundaries, unsigned char *d_image_vector,
                           BaseGridBlock *d_second_simulation_grid, CudaTextureHolder *d_textures,
                           int height_img_boundaries_size, int width_img_boundaries_size,
                           bool yuv_format);
};

#endif //THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
