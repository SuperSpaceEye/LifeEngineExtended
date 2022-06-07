//
// Created by spaceeye on 05.06.22.
//

#include "cuda_image_creator.cuh"

//TODO how the FUCK do i setup a shared lookup table.

struct dcolor {
    unsigned char r{0};
    unsigned char g{0};
    unsigned char b{0};
};

#define BLACK1 dcolor{14,  19,  24}
#define BLACK2 dcolor{56,  62,  77}
#define GRAY1  dcolor{182, 193, 234}
#define GRAY2  dcolor{161, 172, 209}
#define GRAY3  dcolor{167, 177, 215}


__device__ void set_image_pixel(int x, int y, int width, dcolor color, unsigned char * image_vector) {
    auto index = 4 * (y * width + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

__device__ dcolor get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale) {
    int x;
    int y;
    int temp;

    switch (type) {
        case EmptyBlock :
            return dcolor{14, 19, 24};
        case MouthBlock:
            return dcolor{222, 177, 77};
        case ProducerBlock:
            return dcolor{21, 222, 89};
        case MoverBlock:
            return dcolor{96, 212, 255};
        case KillerBlock:
            return dcolor{248, 35, 128};
        case ArmorBlock:
            return dcolor{114, 48, 219};
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
            dcolor rawEyeTexture[5 * 5] = {GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                           GRAY1, GRAY3, BLACK2, GRAY3, GRAY1,
                                           GRAY1, GRAY1, GRAY1, GRAY1, GRAY1};
            return rawEyeTexture[x + y * 5];
        }
        case FoodBlock:     return dcolor{47, 122,183};
        case WallBlock:     return dcolor{128, 128, 128};
        default: return dcolor{14, 19, 24};
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

    dcolor pixel_color = {0,100,0};

    auto w_b = d_width_img_boundaries[x_pos];
    auto h_b = d_height_img_boundaries[y_pos];

    for (int x = w_b.start; x < w_b.stop; x++) {
        for (int y = h_b.start; y < h_b.stop; y++) {
            auto &block = d_second_simulation_grid[d_lin_width[x] + d_lin_height[y] * simulation_height];
            if (d_lin_width[x] < 0 ||
                d_lin_width[x] >= simulation_width ||
                d_lin_height[y] < 0 ||
                d_lin_height[y] >= simulation_height) {
                pixel_color = dcolor{58, 75, 104};
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
                                         std::vector<int> &lin_height, std::vector<unsigned char> &image_vector,
                                         ColorContainer &color_container, EngineDataContainer &dc, int block_size,
                                         std::vector<int> & truncated_lin_width,
                                         std::vector<int> & truncated_lin_height) {
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

    int num_blocks = (width_img_boundaries.size()*height_img_boundaries.size() + block_size - 1) / block_size;


    //WHY WHY WHY THE FUCK vy OF GRID NEEDS TO BE height_img_boundaries.size() WHEN vx IS
    //width_img_boundaries.size() / block_size + 1 WHYYYYYYYYYYYYYYYYYY
    //IT DOESNT MAKE SENSE
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

