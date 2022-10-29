//
// Created by spaceeye on 31.07.22.
//

#include "ImageCreation.h"

void ImageCreation::calculate_linspace(std::vector<int> &lin_width, std::vector<int> &lin_height, int start_x,
                                       int end_x, int start_y, int end_y, int image_width, int image_height) {
    lin_width  = linspace<int>(start_x, end_x, image_width);
    lin_height = linspace<int>(start_y, end_y, image_height);
}

void ImageCreation::calculate_truncated_linspace(int image_width, int image_height,
                                                 const std::vector<int> &lin_width,
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

void ImageCreation::create_image(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                                 uint32_t simulation_width, uint32_t simulation_height, const ColorContainer &cc,
                                 const TexturesContainer &textures, int image_width, int image_height,
                                 std::vector<unsigned char> &image_vector,
                                 const std::vector<BaseGridBlock> &second_grid, bool use_cuda, bool cuda_is_available,
                                 void *cuda_creator_ptr, const std::vector<int> &truncated_lin_width,
                                 const std::vector<int> &truncated_lin_height, bool yuv_format) {
#ifdef __CUDA_USED__
    if (!use_cuda || !cuda_is_available) {
#endif
        ImageCreation::ImageCreationTools::complex_image_creation(lin_width, lin_height, simulation_width,
                                                                  simulation_height, cc, textures, image_width,
                                                                  image_vector, second_grid);
#ifdef __CUDA_USED__
    } else {
        std::vector<Differences> differences{};
        reinterpret_cast<CUDAImageCreator *>(cuda_creator_ptr)->compile_differences(truncated_lin_width, truncated_lin_height,
                                                                                    differences, simulation_width,
                                                                                    simulation_height, second_grid);

        reinterpret_cast<CUDAImageCreator *>(cuda_creator_ptr)->cuda_create_image(image_width, image_height,
                                                                                  lin_width, lin_height, image_vector,
                                                                                  cc, 32, simulation_width,
                                                                                  simulation_height,
                                                                                  differences, yuv_format);
    }
#endif
}

const color &ImageCreation::ImageCreationTools::get_texture_color(BlockTypes type, Rotation rotation,
                                                                  double rxs, double rys,
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

            rxs = -rxs;
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

            rys = -rys;
            rxs += 0.5;
            rys += 0.5;
            break;
    }

    int x = rxs * holder.width;
    int y = rys * holder.height;

    if (x == holder.width) {x--;}
    if (y == holder.height) {y--;}

    return holder.texture[x + y * holder.width];
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

//First calculates what world blocks are seen, then calculates how much of each world block is seen in the frame.
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
    for (int y = 0; y < lin_height.size(); y++) {
        if (last < lin_height[y]) {
            height_img_boundaries.emplace_back(count, y);
            last = lin_height[y];
            count = y;
        }
    }
    height_img_boundaries.emplace_back(count, lin_height.size());

    color pixel_color;
    //width of boundaries of an organisms

    //width bound, height bound
    //goes through seen blocks
    for (auto &w_b: width_img_boundaries) {
        for (auto &h_b: height_img_boundaries) {
            //calculates texture in seen block.
            for (int x = w_b.x; x < w_b.y; x++) {
                for (int y = h_b.x; y < h_b.y; y++) {
                    if (lin_width[x] < 0 ||
                        lin_width[x] >= simulation_width ||
                        lin_height[y] < 0 ||
                        lin_height[y] >= simulation_height) {
                        pixel_color = cc.simulation_background_color;
                    } else {
                        auto &block = second_grid[lin_width[x] + lin_height[y] * simulation_width];
                        //double({pos} - {dim}_b.x) / ({dim}_b.y - {dim}_b.x)
                        // first,  calculate relative position of a pixel inside a texture block.
                        // second, calculate a dimension of a pixel that is going to be displayed.
                        // third,  normalize relative position between 0 and 1 by dividing result of first stage by second one.
                        pixel_color = get_texture_color(block.type,
                                                        block.rotation,
                                                        double(x - w_b.x) / (w_b.y - w_b.x),
                                                        double(y - h_b.x) / (h_b.y - h_b.x),
                                                        textures);
                    }
                    set_image_pixel(x, y, image_width, pixel_color, image_vector);
                }
            }
        }
    }
}