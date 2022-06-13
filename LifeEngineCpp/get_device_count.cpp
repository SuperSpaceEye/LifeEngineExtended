//
// Created by SpaceEye on 12.06.22.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "get_device_count.cuh"

int get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}