//)
// Created by spaceeye on 26.05.2022.
//

#include "SimulationEngineCuda.cuh"

SimulationEngineCuda::SimulationEngineCuda(int block_dim):
    block_dim(block_dim) {

}

//__global__ static void add_vectors(int* d_vec_a, int* d_vec_b, int n) {
//
////    int blockSize = 256;
////    int numBlocks = (N + blockSize - 1) / blockSize;
////    add<<<numBlocks, blockSize>>>(N, x, y);
//
//    //https://developer.nvidia.com/blog/even-easier-introduction-cuda/
//    //int index = blockIdx.x * blockDim.x + threadIdx.x;
//    //  int stride = blockDim.x * gridDim.x;
//    //  for (int i = index; i < n; i += stride)
//    //    y[i] = x[i] + y[i];
//    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//    for (int i = 0; i < n; i++) {
//        d_vec_a[tid] = d_vec_a[tid] + d_vec_b[tid];
//    }
//}


//void SimulationEngineCuda::cuda_test() {
//    {
//        auto num = 10000005;
//        std::cout << "start values init\n";
//        thrust::host_vector<int> h_vec_a(num, 5);
//        thrust::host_vector<int> h_vec_b(num, 10);
//
//        std::cout << "end values init\n";
//        std::cout << "start device values transfer\n";
//        thrust::device_vector<int> d_vec_a(h_vec_a);
//        thrust::device_vector<int> d_vec_b(h_vec_b);
//        std::cout << "end device values transfer\n";
//
//        auto raw_d_vec_a = thrust::raw_pointer_cast(d_vec_a.data());
//        auto raw_d_vec_b = thrust::raw_pointer_cast(d_vec_b.data());
//
//
//        //https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
////        cudaDeviceProp prop;
////        cudaGetDeviceProperties(&prop , 0);
////        std::cout << prop.
//        //https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html#abstract
//
//        //int minGridSize = 0;
//        //int blockSize = 0;
//        //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add_vectors);
//        //std::cout << minGridSize << " " << blockSize << "\n";
//
//        std::cout << "start calculations\n";
//        //https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
//        //https://docs.nvidia.com/cuda/thrust/index.html
//        //https://developer.nvidia.com/blog/even-easier-introduction-cuda/
//        //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//        //https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/
//        //https://stackoverflow.com/questions/44223306/in-an-elementwise-add-cuda-kernel-why-is-the-stride-blockdim-x-griddim-x
//        //https://oneflow2020.medium.com/how-to-choose-the-grid-size-and-block-size-for-a-cuda-kernel-d1ff1f0a7f92
//        //https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s
//
//        std::cout << num << " " << (num/32)*32 << " " << (num/32+1)*32 << "\n";
//        add_vectors<<<num/32+1, 32>>>(raw_d_vec_a, raw_d_vec_b, 10000);
//
//        thrust::copy(d_vec_a.begin(), d_vec_a.end(), h_vec_a.begin());
//        std::cout << "end calculations\n";
//
//        d_vec_a.clear();
//        d_vec_b.clear();
//
//        std::cout << h_vec_a[0] << "\n";
//
//        std::cout << "start check\n";
//        for (auto & item: h_vec_a) {
//            if (item != h_vec_a[0]) {
//                std::cout << "corrupted " << item << "\n";
//            }
//        }
//        std::cout << "end check\n";
//    }
//}

__global__ void setup_kernel(curandState * state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

int get_grid_num(int n, int block_dim) {
    int result = n/block_dim;
    if (result*block_dim < n) {
        return result+1;
    }
    return result;
}

void SimulationEngineCuda::cuda_tick(int n) {
    curandState* random_states;
    cudaMalloc(&random_states, n*sizeof(curandState));
    setup_kernel<<<get_grid_num(150, 32), 32>>>(random_states, 5);



    cudaFree(random_states);
}