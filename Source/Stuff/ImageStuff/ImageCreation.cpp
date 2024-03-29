//
// Created by spaceeye on 31.07.22.
//

#include "ImageCreation.h"
#include "textures.h"

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
                                 const Textures::TexturesContainer &textures, int image_width, int image_height,
                                 std::vector<unsigned char> &image_vector,
                                 const std::vector<BaseGridBlock> &second_grid, bool use_cuda, bool cuda_is_available,
                                 void *cuda_creator_ptr, const std::vector<int> &truncated_lin_width,
                                 const std::vector<int> &truncated_lin_height, bool yuv_format, int kernel_size) {
#ifdef __CUDA_USED__
    if (!use_cuda || !cuda_is_available) {
#endif
        if (kernel_size == 1) {
            ImageCreation::ImageCreationTools::cpu_image_creation(lin_width, lin_height, simulation_width,
                                                                  simulation_height, cc, textures, image_width,
                                                                  image_vector, second_grid);
        } else {
            ImageCreation::ImageCreationTools::cpu_kernel_image_creation(lin_width, lin_height, simulation_width,
                                                                         simulation_height, cc, textures, image_width,
                                                                         image_vector, second_grid, kernel_size);
        }
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

Textures::color ImageCreation::ImageCreationTools::get_texture_color(BlockTypes type, Rotation rotation,
                                                                     double rxs, double rys,
                                                                     const Textures::TexturesContainer &textures) {
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

//I wrote this code at like 1 am or smth so

// If the number of pixels per texture is small, then the textures will be incorrect for some reason.
// That's why, this function for x,y pos will actually get x_1,y_1 to x_kernel_size,y_kernel_size positions with
// dimension expanded by kernel_size and return, the most used color.
//Will do kernel_size^2
Textures::color ImageCreation::ImageCreationTools::get_kernel_texture_color(BlockTypes type,
                                                                            Rotation rotation,
                                                                            int texture_x, int texture_y,
                                                                            int texture_width, int texture_height,
                                                                            int kernel_size,
                                                                            const Textures::TexturesContainer &textures) {
    auto & holder = textures.textures[static_cast<int>(type)];

    if (holder.width == 1 && holder.height == 1) {return holder.texture[0];}

    //will keep track of each color and how many times it was used.
    std::vector<std::pair<Textures::color, int>> used_colors;

    //x1, x2, x3, ... -> x1_1, x1_2, ..., x1_{kernel_size-1}, x2_1, ...
    for (int _x = texture_x * kernel_size; _x < texture_x * kernel_size + kernel_size; _x++) {
        for (int _y = texture_y * kernel_size; _y < texture_y * kernel_size + kernel_size; _y++) {
            auto _color = get_texture_color(type,
                                           rotation,
                                            double(_x)/(texture_width*kernel_size),
                                            double(_y)/(texture_height*kernel_size),
                                           textures);
            bool is_in = false;
            for (auto & [lcol, num]: used_colors) {
                if (lcol.r == _color.r && lcol.g == _color.g && lcol.b == _color.b) {
                    is_in = true;
                    num++;
                    break;
                }
            }
            if (!is_in) {used_colors.emplace_back(std::pair<Textures::color, int>{_color, 1});}
        }
    }

    Textures::color ret_color{};

    int max = 0;

    //get the most used color.
    for (auto & [_color, num]: used_colors) {
        if (num > max) {
            max = num;
            ret_color = _color;
        }
    }

    return ret_color;
}

// depth * ( y * width + x) + z
// depth * width * y + depth * x + z
void ImageCreation::ImageCreationTools::set_image_pixel(int x,
                                                        int y,
                                                        int image_width,
                                                        const Textures::color &color,
                                                        std::vector<unsigned char> &image_vector) {
    auto index = 4 * (y * image_width + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

void ImageCreation::ImageCreationTools::cpu_image_creation(const std::vector<int> &lin_width,
                                                           const std::vector<int> &lin_height,
                                                           uint32_t simulation_width,
                                                           uint32_t simulation_height,
                                                           const ColorContainer &cc, const Textures::TexturesContainer &textures,
                                                           int image_width, std::vector<unsigned char> &image_vector,
                                                           const std::vector<BaseGridBlock> &second_grid) {
    std::vector<int> width_img_boundaries;
    std::vector<int> height_img_boundaries;

    auto last = lin_width[0];
    auto count = 0;
    for (int x = 0; x < lin_width.size(); x++) {
        if (last < lin_width[x]) {
            width_img_boundaries.emplace_back(x - count);
            last = lin_width[x];
            count = x;
        }
    }
    //TODO here is a blunt method to fix error
    width_img_boundaries.emplace_back(INT32_MAX);

    last = lin_height[0];
    count = 0;
    for (int y = 0; y < lin_height.size(); y++) {
        if (last < lin_height[y]) {
            height_img_boundaries.emplace_back(y - count);
            last = lin_height[y];
            count = y;
        }
    }
    height_img_boundaries.emplace_back(INT32_MAX);

    Textures::color pixel_color;
    //width of boundaries of an organisms

    int texture_x_i = 0;
    int texture_y_i = 0;

    int texture_x = 0;
    int texture_y = 0;

    int texture_width  = width_img_boundaries[0];
    int texture_height = height_img_boundaries[0];

    for (int y = 0; y < lin_height.size(); y++) {
        for (int x = 0; x < lin_width.size(); x++) {
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
                                                double(texture_x) / (texture_width),
                                                double(texture_y) / (texture_height),
                                                textures);
            }
            set_image_pixel(x, y, image_width, pixel_color, image_vector);

            texture_x++;
            if (texture_x+1  > texture_width) {texture_x=0; texture_width = width_img_boundaries[++texture_x_i];}
        }
        texture_x=0; texture_x_i = 0; texture_width = width_img_boundaries[0];
        texture_y++;
        if (texture_y+1> texture_height) {texture_y=0; texture_height = height_img_boundaries[++texture_y_i];}
    }
}

void ImageCreation::ImageCreationTools::cpu_kernel_image_creation(const std::vector<int> &lin_width,
                                                                  const std::vector<int> &lin_height,
                                                                  uint32_t simulation_width,
                                                                  uint32_t simulation_height,
                                                                  const ColorContainer &cc, const Textures::TexturesContainer &textures,
                                                                  int image_width, std::vector<unsigned char> &image_vector,
                                                                  const std::vector<BaseGridBlock> &second_grid,
                                                                  int kernel_size) {
    std::vector<int> width_img_boundaries;
    std::vector<int> height_img_boundaries;

    auto last = lin_width[0];
    auto count = 0;
    for (int x = 0; x < lin_width.size(); x++) {
        if (last < lin_width[x]) {
            width_img_boundaries.emplace_back(x - count);
            last = lin_width[x];
            count = x;
        }
    }
    //TODO here is a blunt method to fix error
    width_img_boundaries.emplace_back(INT32_MAX);

    last = lin_height[0];
    count = 0;
    for (int y = 0; y < lin_height.size(); y++) {
        if (last < lin_height[y]) {
            height_img_boundaries.emplace_back(y - count);
            last = lin_height[y];
            count = y;
        }
    }
    height_img_boundaries.emplace_back(INT32_MAX);

    Textures::color pixel_color;
    //width of boundaries of an organisms

    int texture_x_i = 0;
    int texture_y_i = 0;

    int texture_x = 0;
    int texture_y = 0;

    int texture_width  = width_img_boundaries[0];
    int texture_height = height_img_boundaries[0];

    for (int y = 0; y < lin_height.size(); y++) {
        for (int x = 0; x < lin_width.size(); x++) {
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
                pixel_color = get_kernel_texture_color(block.type,
                                                       block.rotation,
                                                       texture_x, texture_y,
                                                       texture_width, texture_height,
                                                       kernel_size, textures);
            }
            set_image_pixel(x, y, image_width, pixel_color, image_vector);

            texture_x++;
            if (texture_x+1  > texture_width) {texture_x=0; texture_width = width_img_boundaries[++texture_x_i];}
        }
        texture_x=0; texture_x_i = 0; texture_width = width_img_boundaries[0];
        texture_y++;
        if (texture_y+1> texture_height) {texture_y=0; texture_height = height_img_boundaries[++texture_y_i];}
    }
}