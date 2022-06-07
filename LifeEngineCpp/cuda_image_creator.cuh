//
// Created by spaceeye on 05.06.22.
//

#ifndef THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
#define THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>

#include "pix_pos.h"
#include "textures.h"
#include "Containers/CPU/ColorContainer.h"
#include "Containers/CPU/EngineDataContainer.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class CUDAImageCreator {
    int * d_lin_width = nullptr;
    int * d_lin_height = nullptr;

    pix_pos *d_width_img_boundaries = nullptr;
    pix_pos *d_height_img_boundaries = nullptr;
    unsigned char * d_image_vector = nullptr;
    BaseGridBlock * d_second_simulation_grid = nullptr;

//    ColorContainer * dd_color_container;
//    Textures * dd_textures;

    int last_image_width = 0;
    int last_image_height = 0;
    int last_lin_width = 0;
    int last_lin_height = 0;

    int last_width_img_boundaries = 0;
    int last_height_img_boundaries = 0;
    int last_simulation_width = 0;
    int last_simulation_height = 0;

    void image_dimensions_changed(int image_width, int image_height) {
        gpuErrchk(cudaFree(d_image_vector));
        gpuErrchk(cudaMalloc((unsigned char**)&d_image_vector, sizeof(unsigned char) * image_width * image_height * 4));
    }

    void simulation_dimensions_changed(int simulation_width, int simulation_height) {
        gpuErrchk(cudaFree(d_second_simulation_grid));
        gpuErrchk(cudaMallocManaged((BaseGridBlock**)&d_second_simulation_grid, sizeof(BaseGridBlock) * simulation_width * simulation_height));
    }

    void img_boundaries_changed(int width_img_boundaries_size, int height_img_boundaries_size) {
        gpuErrchk(cudaFree(d_width_img_boundaries));
        gpuErrchk(cudaFree(d_height_img_boundaries));
        gpuErrchk(cudaMalloc((pix_pos**)&d_width_img_boundaries,  sizeof(pix_pos) * width_img_boundaries_size));
        gpuErrchk(cudaMalloc((pix_pos**)&d_height_img_boundaries, sizeof(pix_pos) * height_img_boundaries_size));
    }

    void lin_size_changed(int lin_width_size, int lin_height_size) {
        gpuErrchk(cudaFree(d_lin_width));
        gpuErrchk(cudaFree(d_lin_height));
        gpuErrchk(cudaMalloc((int**)&d_lin_width, sizeof(int) * lin_width_size));
        gpuErrchk(cudaMalloc((int**)&d_lin_height, sizeof(int) * lin_height_size));
    }

    void color_container_changed(ColorContainer * color_container) {
//        cudaFree(d_color_container);
//        ColorContainer * ad_color_container;
//        cudaMalloc((ColorContainer**)&dd_color_container, sizeof(ColorContainer));
//        cudaMemcpy(dd_color_container,
//                   color_container,
//                   sizeof(ColorContainer),
//                   cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol("d_color_container",
//                           &ad_color_container,
//                           sizeof(ColorContainer),
//                           size_t(color_container),
//                           cudaMemcpyHostToDevice);
    }

    void textures_changed(Textures * textures) {
//        Textures * ad_textures;
//        cudaMalloc((Textures**)&dd_textures, sizeof(Textures));
//        cudaMemcpy(dd_textures,
//                   textures,
//                   sizeof(Textures),
//                   cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol("d_textures",
//                           &ad_textures,
//                           sizeof(Textures),
//                           size_t(textures),
//                           cudaMemcpyHostToDevice);
    }

    void init(int image_width, int image_height,
              int simulation_width, int simulation_height,
              int width_img_boundaries_size, int height_img_boundaries_size,
              int lin_width_size, int lin_height_size,
              ColorContainer * color_container, Textures * textures) {
        image_dimensions_changed(image_width, image_height);
        simulation_dimensions_changed(simulation_width, simulation_height);
        lin_size_changed(lin_width_size, lin_height_size);
//        color_container_changed(color_container);
//        textures_changed(textures);
    }

    void check_if_changed(int image_width, int image_height,
                          int simulation_width, int simulation_height,
                          int width_img_boundaries_size, int height_img_boundaries_size,
                          int lin_width_size, int lin_height_size) {
        if (image_width != last_image_width || image_height != last_image_height) {
            last_image_width = image_width;
            last_image_height = image_height;
            image_dimensions_changed(image_width, image_height);
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
    }

    void copy_to_device(std::vector<int> &lin_width, std::vector<int> &lin_height,
                        std::vector<pix_pos> &width_img_boundaries, std::vector<pix_pos> &height_img_boundaries,
                        std::vector<int> & truncated_lin_width,
                        std::vector<int> & truncated_lin_height,
                        EngineDataContainer &dc) {
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
                   sizeof (pix_pos)*width_img_boundaries.size(),
                   cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_height_img_boundaries,
                   height_img_boundaries.data(),
                   sizeof (pix_pos)*height_img_boundaries.size(),
                   cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_second_simulation_grid,
                   dc.second_simulation_grid.data(),
                   sizeof (BaseGridBlock)*dc.second_simulation_grid.size(),
                   cudaMemcpyHostToDevice))

//        for (auto & width: truncated_lin_width) {
//            for (auto & height: truncated_lin_height) {
//                if (height < 0 || width < 0) { continue;}
//                d_second_simulation_grid[width + height * dc.simulation_height] = dc.second_simulation_grid[width + height * dc.simulation_height];
//            }
//        }

    }

    void copy_result_image(std::vector<unsigned char> &image_vector, int image_width, int image_height) {
        gpuErrchk(cudaMemcpy(image_vector.data(),
                   d_image_vector,
                   sizeof(unsigned char)*image_width*image_height*4,
                   cudaMemcpyDeviceToHost));
    }

public:
    CUDAImageCreator()=default;
    CUDAImageCreator(ColorContainer * color_container, Textures * textures) {
        color_container_changed(color_container);
        textures_changed(textures);
    }

    void free() {
        cudaFree(d_image_vector);
        cudaFree(d_second_simulation_grid);
        cudaFree(d_width_img_boundaries);
        cudaFree(d_height_img_boundaries);
        cudaFree(d_lin_width);
        cudaFree(d_lin_height);
//        cudaFree(d_textures);
//        cudaFree(d_color_container);

        d_lin_width = nullptr;
        d_lin_height = nullptr;
        d_width_img_boundaries = nullptr;
        d_height_img_boundaries = nullptr;
        d_image_vector = nullptr;
        d_second_simulation_grid = nullptr;
//        d_color_container = nullptr;
//        d_textures = nullptr;

        last_image_width = 0;
        last_image_height = 0;
        last_lin_width = 0;
        last_lin_height = 0;

        last_width_img_boundaries = 0;
        last_height_img_boundaries = 0;
        last_simulation_width = 0;
        last_simulation_height = 0;
    }

    void cuda_create_image(int image_width, int image_height,
                           std::vector<int> &lin_width, std::vector<int> &lin_height,
                           std::vector<unsigned char> &image_vector,
                           ColorContainer &color_container, EngineDataContainer &dc, int block_size,
                           std::vector<int> & truncated_lin_width,
                           std::vector<int> & truncated_lin_height);
};

#endif //THELIFEENGINECPP_CUDA_IMAGE_CREATOR_CUH
