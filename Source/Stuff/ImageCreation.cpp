//
// Created by spaceeye on 31.07.22.
//

#include "ImageCreation.h"

void ImageCreation::calculate_linspace(std::vector<int> &lin_width, std::vector<int> &lin_height, int start_x,
                                       int end_x, int start_y, int end_y, int image_width, int image_height) {
    lin_width  = linspace<int>(start_x, end_x, image_width);
    lin_height = linspace<int>(start_y, end_y, image_height);

    //when zoomed, boundaries of simulation grid are more than could be displayed by 1, so we need to delete the last
    // n pixels
    int max_x = lin_width[lin_width.size()-1];
    int max_y = lin_height[lin_height.size()-1];
    int del_x = 0;
    int del_y = 0;
    for (int x = lin_width.size() -1; lin_width[x]  == max_x; x--) {del_x++;}
    for (int y = lin_height.size()-1; lin_height[y] == max_y; y--) {del_y++;}

    for (int i = 0; i < del_x; i++) {lin_width.pop_back();}
    for (int i = 0; i < del_y; i++) {lin_height.pop_back();}
}

void ImageCreation::calculate_truncated_linspace(int image_width, int image_height, const std::vector<int> &lin_width,
                                                 const std::vector<int> &lin_height,
                                                 std::vector<int> &truncated_lin_width,
                                                 std::vector<int> &truncated_lin_height) {
    int min_val = INT32_MIN;
    for (int x = 0; x < image_width; x++) {if (lin_width[x] > min_val) {min_val = lin_width[x]; truncated_lin_width.push_back(min_val);}}
    truncated_lin_width.pop_back();
    min_val = INT32_MIN;
    for (int y = 0; y < image_height; y++) {if (lin_height[y] > min_val) {min_val = lin_height[y]; truncated_lin_height.push_back(min_val);}}
    truncated_lin_height.pop_back();
}

const color &ImageCreation::ImageCreationTools::get_texture_color(BlockTypes type, Rotation rotation, float rxs,
                                                                  float rys,
                                                                  const TexturesContainer &textures) {
    auto & holder = textures.textures[static_cast<int>(type)];

    if (holder.width == 1 && holder.height == 1) {return holder.texture[0];}

    switch (rotation) {
        case Rotation::UP:
            break;
        case Rotation::LEFT:
            rxs -= 0.5;
            rys -= 0.5;

            std::swap(rxs, rys);

            rys = -rys;
            rxs += 0.5;
            rys += 0.5;
            break;
        case Rotation::DOWN:
            rxs -= 0.5;
            rys -= 0.5;

            rxs = -rxs;
            rys = -rys;

            rxs += 0.5;
            rys += 0.5;
            break;
        case Rotation::RIGHT:
            rxs -= 0.5;
            rys -= 0.5;

            std::swap(rxs, rys);

            rxs = -rxs;
            rxs += 0.5;
            rys += 0.5;
            break;
    }

    int x = rxs * holder.width;
    int y = rys * holder.height;

    if (x == holder.width) {x--;}
    if (y == holder.height) {y--;}

    return holder.texture.at(x + y * holder.width);
}

// depth * ( y * width + x) + z
// depth * width * y + depth * x + z
void ImageCreation::ImageCreationTools::set_image_pixel(int x,
                                                        int y,
                                                        int image_width,
                                                        const color &color,
                                                        std::vector<unsigned char> &image_vector) {
    auto index = 4 * (y * image_width + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

void ImageCreation::ImageCreationTools::complex_image_creation(const std::vector<int> &lin_width,
                                                               const std::vector<int> &lin_height,
                                                               uint32_t simulation_width,
                                                               uint32_t simulation_height,
                                                               const ColorContainer &cc, const TexturesContainer &textures,
                                                               int image_width, std::vector<unsigned char> &image_vector,
                                                               const std::vector<BaseGridBlock> &second_grid) {
    //x - start, y - stop
    std::vector<Vector2<int>> width_img_boundaries;
    std::vector<Vector2<int>> height_img_boundaries;

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

    color pixel_color;
    //width of boundaries of an organisms

    //width bound, height bound
    for (auto &w_b: width_img_boundaries) {
        for (auto &h_b: height_img_boundaries) {
            for (int x = w_b.x; x < w_b.y; x++) {
                for (int y = h_b.x; y < h_b.y; y++) {
                    if (lin_width[x] < 0 ||
                        lin_width[x] >= simulation_width ||
                        lin_height[y] < 0 ||
                        lin_height[y] >= simulation_height) {
                        pixel_color = cc.simulation_background_color;
                    } else {
                        auto &block = second_grid[lin_width[x] + lin_height[y] * simulation_width];
                        pixel_color = get_texture_color(block.type,
                                                        block.rotation,
                                                        float(x - w_b.x) / (w_b.y - w_b.x),
                                                        float(y - h_b.x) / (h_b.y - h_b.x),
                                                        textures);
                    }
                    set_image_pixel(x, y, image_width, pixel_color, image_vector);
                }
            }
        }
    }
}