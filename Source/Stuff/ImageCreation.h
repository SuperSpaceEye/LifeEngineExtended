//
// Created by spaceeye on 31.07.22.
//

#ifndef LIFEENGINEEXTENDED_IMAGECREATION_H
#define LIFEENGINEEXTENDED_IMAGECREATION_H

#include <vector>
#include <cstdint>

#include "../Containers/CPU/EngineDataContainer.h"
#include "../Containers/CPU/ColorContainer.h"

#include "Linspace.h"
#include "Vector2.h"
#include "textures.h"

#if __CUDA_USED__
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
                      const TexturesContainer &textures, int image_width, int image_height,
                      std::vector<unsigned char> &image_vector,
                      const std::vector<BaseGridBlock> &second_grid, bool use_cuda, bool cuda_is_available,
                      void *cuda_creator_ptr, const std::vector<int> &truncated_lin_width,
                      const std::vector<int> &truncated_lin_height, bool cuda_yuv_format);

    namespace ImageCreationTools {
        const color &get_texture_color(BlockTypes type,
                                       Rotation rotation,
                                       double rxs,
                                       double rys,
                                       const TexturesContainer &textures);

        void set_image_pixel(int x,
                             int y,
                             int image_width,
                             const color &color,
                             std::vector<unsigned char> &image_vector);

        void complex_image_creation(const std::vector<int> &lin_width,
                                    const std::vector<int> &lin_height,
                                    uint32_t simulation_width,
                                    uint32_t simulation_height,
                                    const ColorContainer &cc,
                                    const TexturesContainer &textures,
                                    int image_width,
                                    std::vector<unsigned char> &image_vector,
                                    const std::vector<BaseGridBlock> &second_grid);
    }
}

#endif //LIFEENGINEEXTENDED_IMAGECREATION_H
