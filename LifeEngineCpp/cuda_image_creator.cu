//
// Created by spaceeye on 05.06.22.
//

#include "cuda_image_creator.cuh"

#define BLACK1 color{14,  19,  24}
#define BLACK2 color{56,  62,  77}
#define GRAY1  color{182, 193, 234}
#define GRAY2  color{161, 172, 209}
#define GRAY3  color{167, 177, 215}

__constant__ ColorContainer const_color_container;

void CUDAImageCreator::load_symbols(ColorContainer * colorContainer) {
    gpuErrchk(cudaMemcpyToSymbol(const_color_container, colorContainer, sizeof(ColorContainer)));
}

__device__ void set_image_pixel(int x, int y, int width, color color, unsigned char * image_vector) {
    auto index = 4 * (y * width + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

__device__ color get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale) {
    int x;
    int y;
    int temp;

    switch (type) {
        case EmptyBlock :
            return const_color_container.empty_block;
        case MouthBlock:
            return const_color_container.mouth;
        case ProducerBlock:
            return const_color_container.producer;
        case MoverBlock:
            return const_color_container.mover;
        case KillerBlock:
            return const_color_container.killer;
        case ArmorBlock:
            return const_color_container.armor;
        case EyeBlock: {
            x = relative_x_scale * 5;
            y = relative_y_scale * 5;
            {
                switch (rotation) {
                    case Rotation::UP:
                        break;
                    case Rotation::LEFT:
                        x -= 2;
                        y -= 2;

                        temp = x;
                        x = y;
                        y = temp;

                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::DOWN:
                        x -= 2;
                        y -= 2;
                        x = -x;
                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::RIGHT:
                        x -= 2;
                        y -= 2;

                        temp = x;
                        x = y;
                        y = temp;

                        x = -x;
                        x += 2;
                        y += 2;
                        break;
                }
            }
            color rawEyeTexture[5 * 5] = {GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY3, BLACK2, GRAY3, GRAY1,
                                           GRAY1, GRAY1, GRAY1, GRAY1, GRAY1};
            return rawEyeTexture[x + y * 5];
        }
        case FoodBlock:     return const_color_container.food;
        case WallBlock:     return const_color_container.wall;
        default: return const_color_container.simulation_background_color;
    }
}

__global__ void create_image_kernel(int image_width, int simulation_width, int simulation_height, int width_img_size, int height_img_size,
                                           int * d_lin_width, int * d_lin_height,
                                           pix_pos * d_width_img_boundaries, pix_pos * d_height_img_boundaries,
                                           unsigned char * d_image_vector,
                                           BaseGridBlock * d_second_simulation_grid
                                           ) {
    auto x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    auto y_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_pos >= width_img_size || y_pos >= height_img_size) {return;}

    color pixel_color;

    auto w_b = d_width_img_boundaries[x_pos];
    auto h_b = d_height_img_boundaries[y_pos];

    for (int x = w_b.start; x < w_b.stop; x++) {
        for (int y = h_b.start; y < h_b.stop; y++) {
            auto &block = d_second_simulation_grid[d_lin_width[x] + d_lin_height[y] * simulation_width];
            if (d_lin_width[x] < 0 ||
                d_lin_width[x] >= simulation_width ||
                d_lin_height[y] < 0 ||
                d_lin_height[y] >= simulation_height) {
                pixel_color = const_color_container.simulation_background_color;
            } else {
                pixel_color = get_texture_color(block.type,
                                                block.rotation,
                                                float(x - w_b.start) / (w_b.stop - w_b.start),
                                                float(y - h_b.start) / (h_b.stop - h_b.start)
                                                );
            }
            set_image_pixel(x, y, image_width, pixel_color, d_image_vector);
        }
    }
}

void CUDAImageCreator::cuda_create_image(int image_width, int image_height, std::vector<int> &lin_width,
                                         std::vector<int> &lin_height,
                                         std::vector<unsigned char> &image_vector, ColorContainer &color_container,
                                         EngineDataContainer &dc, int block_size, std::vector<int> &truncated_lin_width,
                                         std::vector<int> &truncated_lin_height) {
    std::vector<pix_pos> width_img_boundaries;
    std::vector<pix_pos> height_img_boundaries;

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

    check_if_changed(image_width, image_height,
                     dc.simulation_width, dc.simulation_height,
                     width_img_boundaries.size(), height_img_boundaries.size(),
                     lin_width.size(), lin_height.size());

    copy_to_device(lin_width, lin_height, width_img_boundaries, height_img_boundaries, truncated_lin_width, truncated_lin_height, dc);

    load_symbols(&color_container);

    dim3 grid((width_img_boundaries.size() + block_size - 1) / block_size,
              height_img_boundaries.size());

    dim3 block(block_size, block_size);

    create_image_kernel<<<grid, block_size>>>(image_width,
                                                  dc.simulation_width, dc.simulation_height,
                                                  width_img_boundaries.size(),height_img_boundaries.size(),
                                                  d_lin_width, d_lin_height,
                                                  d_width_img_boundaries, d_height_img_boundaries,
                                                  d_image_vector, d_second_simulation_grid
//                                                  dd_color_container, dd_textures
                                                  );

    gpuErrchk( cudaDeviceSynchronize() );

    copy_result_image(image_vector, image_width, image_height);
}

