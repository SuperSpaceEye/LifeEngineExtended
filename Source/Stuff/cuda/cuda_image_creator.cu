//
// Created by spaceeye on 05.06.22.
//
#define CUDA_IMAGE_CREATOR_EXPORTS

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_image_creator.cuh"

#define BLACK1 color{14,  19,  24}
#define BLACK2 color{56,  62,  77}
#define GRAY1  color{182, 193, 234}
#define GRAY2  color{161, 172, 209}
#define GRAY3  color{167, 177, 215}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__constant__ ColorContainer const_color_container;

void CUDAImageCreator::load_symbols(const ColorContainer *colorContainer) {
    gpuErrchk(cudaMemcpyToSymbol(const_color_container, colorContainer, sizeof(ColorContainer)));
}

__device__ void set_image_pixel(int x, int y, int width, color color, unsigned char * image_vector) {
    auto index = 4 * (y * width + x);
    image_vector[index+2] = color.r;
    image_vector[index+1] = color.g;
    image_vector[index  ] = color.b;
}

//https://stackoverflow.com/questions/1737726/how-to-perform-rgb-yuv-conversion-in-c-c
#define CLIP(X) ( (X) > 255 ? 255 : (X) < 0 ? 0 : (X))

// RGB -> YUV
#define RGB2Y(R, G, B) CLIP(( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16)
#define RGB2U(R, G, B) CLIP(( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128)
#define RGB2V(R, G, B) CLIP(( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128)
//https://stackoverflow.com/questions/27822017/planar-yuv420-data-layout
__device__ void set_yuv_image_pixel(int x, int y, int width, int height, color color, unsigned char *image_vector) {
    //y
    image_vector[y * width + x] = RGB2Y(color.r, color.g, color.b);
    //u
    image_vector[(y / 2) * (width / 2) + (x / 2) + width*height] = RGB2U(color.r, color.g, color.b);
    //v
    image_vector[(y / 2) * (width / 2) + (x / 2) + width*height + width*height / 4] = RGB2V(color.r, color.g, color.b);
}

__device__ color get_texture_color(BlockTypes type, Rotation rotation, float rxs, float rys, CudaTextureHolder * textures) {
    auto & holder = textures[static_cast<int>(type)];

    if (holder.width == 1 && holder.height == 1) {return holder.texture[0];}

    float temp;

    switch (rotation) {
        case Rotation::UP:
            break;
        case Rotation::LEFT:
            rxs -= 0.5;
            rys -= 0.5;

            temp = rxs;
            rxs = rys;
            rys = temp;

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

            temp = rxs;
            rxs = rys;
            rys = temp;

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

__global__ void create_image_kernel(int image_width, int simulation_width, int simulation_height, int width_img_size,
                                    int height_img_size, int *d_lin_width, int *d_lin_height,
                                    Vector2<int> *d_width_img_boundaries,
                                    Vector2<int> *d_height_img_boundaries, unsigned char *d_image_vector,
                                    BaseGridBlock *d_second_simulation_grid, CudaTextureHolder *textures) {
    auto x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    auto y_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_pos >= width_img_size || y_pos >= height_img_size) {return;}

    color pixel_color;

    //x - start, y - stop

    auto w_b = d_width_img_boundaries[x_pos];
    auto h_b = d_height_img_boundaries[y_pos];

    for (int y = h_b.x; y < h_b.y; y++) {
        for (int x = w_b.x; x < w_b.y; x++) {
            auto &block = d_second_simulation_grid[d_lin_width[x] + d_lin_height[y] * simulation_width];

            if (d_lin_width[x] < 0 ||
                d_lin_width[x] >= simulation_width ||
                d_lin_height[y] < 0 ||
                d_lin_height[y] >= simulation_height) {
                pixel_color = const_color_container.simulation_background_color;
            } else {
                pixel_color = get_texture_color(block.type,
                                                block.rotation,
                                                float(x - w_b.x) / (w_b.y - w_b.x),
                                                float(y - h_b.x) / (h_b.y - h_b.x),
                                                textures
                );
            }
            set_image_pixel(x, y, image_width, pixel_color, d_image_vector);
        }
    }
}

__global__ void
create_yuv_image_kernel(int image_width, int image_height, int simulation_width, int simulation_height,
                        int width_img_size, int height_img_size, int *d_lin_width, int *d_lin_height,
                        Vector2<int> *d_width_img_boundaries, Vector2<int> *d_height_img_boundaries,
                        unsigned char *d_image_vector, BaseGridBlock *d_second_simulation_grid,
                        CudaTextureHolder *textures) {
    auto x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    auto y_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_pos >= width_img_size || y_pos >= height_img_size) {return;}

    color pixel_color;

    //x - start, y - stop

    auto w_b = d_width_img_boundaries[x_pos];
    auto h_b = d_height_img_boundaries[y_pos];

    for (int y = h_b.x; y < h_b.y; y++) {
        for (int x = w_b.x; x < w_b.y; x++) {
            auto &block = d_second_simulation_grid[d_lin_width[x] + d_lin_height[y] * simulation_width];

            if (d_lin_width[x] < 0 ||
                d_lin_width[x] >= simulation_width ||
                d_lin_height[y] < 0 ||
                d_lin_height[y] >= simulation_height) {
                pixel_color = const_color_container.simulation_background_color;
            } else {
                pixel_color = get_texture_color(block.type,
                                                block.rotation,
                                                float(x - w_b.x) / (w_b.y - w_b.x),
                                                float(y - h_b.x) / (h_b.y - h_b.x),
                                                textures
                );
            }
            set_yuv_image_pixel(x, y, image_width, image_height, pixel_color, d_image_vector);
        }
    }
}

__global__ void
update_differences(int simulation_width, int diff_len, Differences *diffs, BaseGridBlock *d_second_simulation_grid) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= diff_len) { return;}
    auto diff = diffs[i_pos];
    d_second_simulation_grid[diff.x + diff.y * simulation_width].type = diff.type;
    d_second_simulation_grid[diff.x + diff.y * simulation_width].rotation = diff.rotation;
}

void
CUDAImageCreator::compile_differences(const std::vector<int> &truncated_lin_width, const std::vector<int> &truncated_lin_height,
                                      std::vector<Differences> &host_differences, int simulation_width,
                                      int simulation_height, const std::vector<BaseGridBlock> &simple_state_grid) {
    creating_image = true;
    if (do_not_create_image) { creating_image = false; return;}
    std::atomic_thread_fence(std::memory_order_seq_cst);
    host_differences.reserve(truncated_lin_width.size() * truncated_lin_height.size());
    if (device_state_grid.size() != simple_state_grid.size()) {
        device_state_grid.resize(simple_state_grid.size());
    }

    for (auto y: truncated_lin_height) {
        if (y < 0 || y >= simulation_height) { continue;}
        for (auto x: truncated_lin_width) {
            if (x < 0 || x >= simulation_width) { continue;}
            auto & hb = simple_state_grid[x + y * simulation_width];
            auto & db = device_state_grid[x + y * simulation_width];

            //TODO different vectors for rotation and type to save time on copying?
            if (hb.type != db.type || hb.rotation != db.rotation) {
                db.type     = hb.type;
                db.rotation = hb.rotation;

                host_differences.emplace_back(Differences{static_cast<uint32_t>(x),
                                                          static_cast<uint32_t>(y),
                                                          hb.type,
                                                          hb.rotation,
                });
            }
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    creating_image = false;
}

void CUDAImageCreator::cuda_create_image(int image_width, int image_height, const std::vector<int> &lin_width,
                                         const std::vector<int> &lin_height, std::vector<unsigned char> &image_vector,
                                         const ColorContainer &color_container, int block_size, int simulation_width,
                                         int simulation_height, std::vector<Differences> &differences,
                                         bool yuv_format) {
    if (do_not_create_image) { return;}
    creating_image = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::vector<Vector2<int>> width_img_boundaries;
    std::vector<Vector2<int>> height_img_boundaries;

    //TODO It shouldn't be too heavy on performance, but maybe only do if the viewpoint changes. Same for lin_{dimension}
    auto last = lin_width[0];
    auto count = 0;
    for (int x = 0; x < lin_width.size(); x++) {
        if (last < lin_width[x]) {
            width_img_boundaries.emplace_back(count, x);
            last = lin_width[x];
            count = x;
        }
    }
    width_img_boundaries.emplace_back(count, lin_width.size());

    last = lin_height[0];
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
                     simulation_width, simulation_height,
                     width_img_boundaries.size(), height_img_boundaries.size(),
                     lin_width.size(), lin_height.size(), differences.size(), yuv_format);

    copy_to_device(lin_width, lin_height, width_img_boundaries, height_img_boundaries, differences);

    update_differences<<<(differences.size() + block_size - 1) / block_size, block_size>>>(
            simulation_width, differences.size(),
            d_differences,
            d_second_simulation_grid);

    gpuErrchk( cudaDeviceSynchronize() );

    load_symbols(&color_container);

    cuda_call_create_image(image_width, image_height, image_vector, block_size, simulation_width, simulation_height,
                           d_lin_width, d_lin_height, d_width_img_boundaries,
                           d_height_img_boundaries, d_image_vector, d_second_simulation_grid, d_textures,
                           height_img_boundaries.size(), width_img_boundaries.size(), yuv_format);

    std::atomic_thread_fence(std::memory_order_seq_cst);
    creating_image = false;
}

void
CUDAImageCreator::cuda_call_create_image(int image_width, int image_height, std::vector<unsigned char> &image_vector,
                                         int block_size, int simulation_width, int simulation_height, int *d_lin_width,
                                         int *d_lin_height, Vector2<int> *d_width_img_boundaries,
                                         Vector2<int> *d_height_img_boundaries, unsigned char *d_image_vector,
                                         BaseGridBlock *d_second_simulation_grid, CudaTextureHolder *d_textures,
                                         int height_img_boundaries_size, int width_img_boundaries_size,
                                         bool yuv_format) {
    dim3 grid((width_img_boundaries_size + block_size - 1) / block_size,
              height_img_boundaries_size);

    if (yuv_format) {
        create_yuv_image_kernel<<<grid, block_size>>>(image_width,
                                                      image_height,
                                                      simulation_width, simulation_height,
                                                      width_img_boundaries_size, height_img_boundaries_size,
                                                      d_lin_width, d_lin_height,
                                                      d_width_img_boundaries, d_height_img_boundaries,
                                                      d_image_vector, d_second_simulation_grid, d_textures);

    } else {
        create_image_kernel<<<grid, block_size>>>(image_width,
                                                  simulation_width, simulation_height,
                                                  width_img_boundaries_size, height_img_boundaries_size,
                                                  d_lin_width, d_lin_height,
                                                  d_width_img_boundaries, d_height_img_boundaries,
                                                  d_image_vector, d_second_simulation_grid, d_textures);
    }

    gpuErrchk(cudaDeviceSynchronize())

    copy_result_image(image_vector, image_width, image_height, d_image_vector, yuv_format);
}

void CUDAImageCreator::free() {
    do_not_create_image = true;
    while (creating_image) {}
    std::atomic_thread_fence(std::memory_order_seq_cst);

    cudaFree(d_image_vector);
    cudaFree(d_second_simulation_grid);
    cudaFree(d_width_img_boundaries);
    cudaFree(d_height_img_boundaries);
    cudaFree(d_lin_width);
    cudaFree(d_lin_height);
    cudaFree(d_differences);
    free_textures();

    device_state_grid.clear();
    device_state_grid = std::vector<BaseGridBlock>{};

    d_lin_width = nullptr;
    d_lin_height = nullptr;
    d_width_img_boundaries = nullptr;
    d_height_img_boundaries = nullptr;
    d_image_vector = nullptr;
    d_second_simulation_grid = nullptr;
    d_differences = nullptr;

    last_image_width = 0;
    last_image_height = 0;
    last_lin_width = 0;
    last_lin_height = 0;

    last_width_img_boundaries = 0;
    last_height_img_boundaries = 0;
    last_simulation_width = 0;
    last_simulation_height = 0;
    last_differences = 0;
}

void CUDAImageCreator::copy_result_image(std::vector<unsigned char> &image_vector, int image_width, int image_height,
                                         unsigned char *d_image_vector, bool yuv_format) {
    int num_elements = 4;
    if (yuv_format) {num_elements = 3;}

    gpuErrchk(cudaMemcpy(image_vector.data(),
                         d_image_vector,
                         sizeof(unsigned char)*image_width*image_height*num_elements,
                         cudaMemcpyDeviceToHost));
}

void CUDAImageCreator::copy_textures(TexturesContainer &container) {
    free_textures();

    std::vector<CudaTextureHolder> temp_container;

    for (auto & texture: container.textures) {
        auto temp_holder = CudaTextureHolder{texture.width, texture.height};

        color * temp_d_texture_pointer = nullptr;

        gpuErrchk(cudaMalloc((color**)&temp_d_texture_pointer, sizeof(color) * texture.width * texture.height));

        gpuErrchk(cudaMemcpy(temp_d_texture_pointer,
                             texture.texture.data(),
                             sizeof(color) * texture.width * texture.height,
                             cudaMemcpyHostToDevice));

        d_textures_pointers.emplace_back(temp_d_texture_pointer);

        temp_holder.texture = temp_d_texture_pointer;
        temp_container.emplace_back(temp_holder);
    }

    gpuErrchk(cudaMalloc((CudaTextureHolder**)&d_textures, sizeof(CudaTextureHolder)*temp_container.size()));

    gpuErrchk(cudaMemcpy(d_textures,
                         temp_container.data(),
                         sizeof(CudaTextureHolder)*temp_container.size(),
                         cudaMemcpyHostToDevice));

    do_not_create_image = false;
}

void CUDAImageCreator::free_textures() {
    for (auto texture_pointer: d_textures_pointers) {
        cudaFree(texture_pointer);
    }
    d_textures_pointers.clear();

    cudaFree(d_textures);
    d_textures = nullptr;
}

void CUDAImageCreator::copy_to_device(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                                      const std::vector<Vector2<int>> &width_img_boundaries,
                                      const std::vector<Vector2<int>> &height_img_boundaries,
                                      const std::vector<Differences> &host_differences) {
    gpuErrchk(cudaMemcpy(d_lin_width,
                         lin_width.data(),
                         sizeof (int)*lin_width.size(),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_lin_height,
                         lin_height.data(),
                         sizeof (int)*lin_height.size(),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_width_img_boundaries,
                         width_img_boundaries.data(),
                         sizeof (Vector2<int>)*width_img_boundaries.size(),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_height_img_boundaries,
                         height_img_boundaries.data(),
                         sizeof (Vector2<int>)*height_img_boundaries.size(),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_differences,
                         host_differences.data(),
                         sizeof(Differences)*host_differences.size(),
                         cudaMemcpyHostToDevice));
}

void CUDAImageCreator::check_if_changed(int image_width, int image_height, int simulation_width, int simulation_height,
                                        int width_img_boundaries_size, int height_img_boundaries_size,
                                        int lin_width_size, int lin_height_size, int differences, bool yuv_format) {
    if (image_width != last_image_width || image_height != last_image_height) {
        last_image_width = image_width;
        last_image_height = image_height;
        image_dimensions_changed(image_width, image_height, yuv_format);
    }
    if (simulation_width != last_simulation_width || simulation_height != last_simulation_height) {
        last_simulation_width = simulation_width;
        last_simulation_height = simulation_height;
        simulation_dimensions_changed(simulation_width, simulation_height);
    }
    if (width_img_boundaries_size != last_width_img_boundaries || height_img_boundaries_size != last_height_img_boundaries) {
        last_width_img_boundaries = width_img_boundaries_size;
        last_height_img_boundaries = height_img_boundaries_size;
        img_boundaries_changed(width_img_boundaries_size, height_img_boundaries_size);
    }
    if (lin_width_size != last_lin_width || lin_height_size != last_lin_height) {
        last_lin_width = lin_width_size;
        last_lin_height = lin_height_size;
        lin_size_changed(lin_width_size, lin_height_size);
    }

    if (differences != last_differences) {
        last_differences = differences;
        differences_changed(differences);
    }
}

void CUDAImageCreator::image_dimensions_changed(int image_width, int image_height, bool yuv_format) {
    gpuErrchk(cudaFree(d_image_vector));
    int modifier = 4;
    if (yuv_format) {modifier = 3;}
    gpuErrchk(cudaMalloc((unsigned char**)&d_image_vector, sizeof(unsigned char) * image_width * image_height * modifier));
}

void CUDAImageCreator::simulation_dimensions_changed(int simulation_width, int simulation_height) {
    gpuErrchk(cudaFree(d_second_simulation_grid));
    gpuErrchk(cudaMallocManaged((BaseGridBlock**)&d_second_simulation_grid, sizeof(BaseGridBlock) * simulation_width * simulation_height));
}

void CUDAImageCreator::img_boundaries_changed(int width_img_boundaries_size, int height_img_boundaries_size) {
    gpuErrchk(cudaFree(d_width_img_boundaries));
    gpuErrchk(cudaFree(d_height_img_boundaries));
    gpuErrchk(cudaMalloc((Vector2<int>**)&d_width_img_boundaries,  sizeof(Vector2<int>) * width_img_boundaries_size));
    gpuErrchk(cudaMalloc((Vector2<int>**)&d_height_img_boundaries, sizeof(Vector2<int>) * height_img_boundaries_size));
}

void CUDAImageCreator::lin_size_changed(int lin_width_size, int lin_height_size) {
    gpuErrchk(cudaFree(d_lin_width));
    gpuErrchk(cudaFree(d_lin_height));
    gpuErrchk(cudaMalloc((int**)&d_lin_width, sizeof(int) * lin_width_size));
    gpuErrchk(cudaMalloc((int**)&d_lin_height, sizeof(int) * lin_height_size));
}

void CUDAImageCreator::differences_changed(int differences) {
    gpuErrchk(cudaFree(d_differences));
    gpuErrchk(cudaMalloc((Differences**)&d_differences, sizeof(Differences)*differences));
}