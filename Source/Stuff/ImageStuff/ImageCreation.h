//
// Created by spaceeye on 31.07.22.
//

#ifndef LIFEENGINEEXTENDED_IMAGECREATION_H
#define LIFEENGINEEXTENDED_IMAGECREATION_H

#include <vector>
#include <cstdint>

#include "Containers/EngineDataContainer.h"
#include "Containers/ColorContainer.h"

#include "Stuff/Linspace.h"
#include "Stuff/structs/Vector2.h"
#include "textures.h"

#ifdef __CUDA_USED__
#include "cuda_image_creator.cuh"
#endif

namespace ImageCreation {
    void calculate_linspace(std::vector<int> & lin_width,
                            std::vector<int> & lin_height,
                            int start_x,
                            int end_x,
                            int start_y,
                            int end_y,
                            int image_width,
                            int image_height);

    void calculate_truncated_linspace(
            int image_width,
            int image_height,
            const std::vector<int> &lin_width,
            const std::vector<int> &lin_height,
            std::vector<int> & truncated_lin_width,
            std::vector<int> & truncated_lin_height);

    void create_image(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                      uint32_t simulation_width, uint32_t simulation_height, const ColorContainer &cc,
                      const Textures::TexturesContainer &textures, int image_width, int image_height,
                      std::vector<unsigned char> &image_vector,
                      const std::vector<BaseGridBlock> &second_grid, bool use_cuda, bool cuda_is_available,
                      void *cuda_creator_ptr, const std::vector<int> &truncated_lin_width,
                      const std::vector<int> &truncated_lin_height, bool yuv_format, int kernel_size);

    namespace ImageCreationTools {
        Textures::color get_texture_color(BlockTypes type,
                                          Rotation rotation,
                                          double texture_x, double texture_y,
                                          double texture_width,double texture_height,
                                          int _, const Textures::TexturesContainer &textures);

        Textures::color get_kernel_texture_color(BlockTypes type,
                                                 Rotation rotation,
                                                 int texture_x, int texture_y,
                                                 int texture_width, int texture_height,
                                                 int kernel_size,
                                                 const Textures::TexturesContainer &textures);

        void set_image_pixel(int x,
                             int y,
                             int image_width,
                             const Textures::color &color,
                             std::vector<unsigned char> &image_vector);
        void prepare_data(std::vector<int> &width_img_boundaries, std::vector<int> &height_img_boundaries,
                          const std::vector<int> &lin_width, const std::vector<int> &lin_height);

        template<auto texture_fn>
        void image_creation(const std::vector<int> &lin_width,
                            const std::vector<int> &lin_height,
                            uint32_t simulation_width,
                            uint32_t simulation_height,
                            const ColorContainer &cc,
                            const Textures::TexturesContainer &textures,
                            int image_width,
                            std::vector<unsigned char> &image_vector,
                            const std::vector<BaseGridBlock> &second_grid,
                            int kernel_size = 0);
    }
}

#endif //LIFEENGINEEXTENDED_IMAGECREATION_H
