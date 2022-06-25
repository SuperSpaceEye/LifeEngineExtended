//
// Created by spaceeye on 26.05.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONENGINECUDA_CUH
#define THELIFEENGINECPP_SIMULATIONENGINECUDA_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "../../Organism/CUDA/CUDA_Organism.cuh"

class SimulationEngineCuda {
public:
    SimulationEngineCuda(int block_dim=32);

    int block_dim = 32;

    thrust::device_vector<CUDA_Organism> cuda_organisms;

    void cuda_tick(int n=150);
};


#endif //THELIFEENGINECPP_SIMULATIONENGINECUDA_CUH
