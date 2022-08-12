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

    namespace ImageCreationTools {
        const color &get_texture_color(BlockTypes type,
                                       Rotation rotation,
                                       float relative_x_scale,
                                       float relative_y_scale,
                                       const ColorContainer &cc,
                                       const Textures &textures);

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
                                    const Textures &textures,
                                    int image_width,
                                    std::vector<unsigned char> &image_vector,
                                    const std::vector<BaseGridBlock> &second_grid);
    }
}

#endif //LIFEENGINEEXTENDED_IMAGECREATION_H
