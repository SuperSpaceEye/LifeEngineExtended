//
// Created by spaceeye on 28.05.2022.
//

#ifndef THELIFEENGINECPP_CUDAENGINEDATACONTAINER_H
#define THELIFEENGINECPP_CUDAENGINEDATACONTAINER_H

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "../../GridBlocks/BaseGridBlock.h"
#include "../../Organism/CUDA/CUDA_Organism.cuh"
#include "../../Containers/CPU/SimulationParameters.h"

struct CUDASharedData {
    uint32_t simulation_width = 600;
    uint32_t simulation_height = 600;
    SimulationParameters sp{};
};

struct CUDAEngineDataContainer {
    CUDASharedData shared_data{};

    thrust::device_vector<CUDA_Organism> d_organisms;
    std::vector<BaseGridBlock> image_grid;
};

#endif //THELIFEENGINECPP_CUDAENGINEDATACONTAINER_H
