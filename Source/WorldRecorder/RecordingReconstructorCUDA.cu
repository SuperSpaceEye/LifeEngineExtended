//
// Created by spaceeye on 25.10.22.
//

#include "RecordingReconstructorCUDA.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <utility>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__
void swap(RecCudaOrganism * org1, RecCudaOrganism * org2) {
    int temp;
    SerializedOrganismBlockContainer * temp_b;

    temp = org1->x;
    org1->x = org2->x;
    org2->x = temp;

    temp = org1->y;
    org1->y = org2->y;
    org2->y = temp;

    temp = org1->vector_index;
    org1->vector_index = org2->vector_index;
    org2->vector_index = temp;

    temp_b = org1->_organism_blocks;
    org1->_organism_blocks = org2->_organism_blocks;
    org2->_organism_blocks = temp_b;
}

struct CVector2 {
    int x;
    int y;
};

__device__
inline CVector2 get_pos(Rotation rotation, int relative_x, int relative_y) {
    switch (rotation) {
        case Rotation::UP:    return CVector2{relative_x, relative_y};
        case Rotation::LEFT:  return CVector2{relative_y, -relative_x};
        case Rotation::DOWN:  return CVector2{-relative_x, -relative_y};
        case Rotation::RIGHT: return CVector2{-relative_y, relative_x};
        default: return CVector2{relative_x, relative_y};
    }
}

__device__
void apply_new_organism_blocks(SerializedOrganismBlockContainer ** cont1,
                               SerializedOrganismBlockContainer ** cont2) {
    SerializedOrganismBlockContainer * temp;
    temp = *cont1;
    *cont1 = *cont2;
    *cont2 = temp;
}

__device__
void apply_new_organism(RecCudaOrganism * org1, RecCudaOrganism * org2) {
    org1->x = org2->x;
    org1->y = org2->y;
    org1->vector_index = org2->vector_index;

    apply_new_organism_blocks(&org1->_organism_blocks, &org2->_organism_blocks);

    cudaFree(org2->_organism_blocks);
    cudaFree(org2);
}

void RecCudaOrganism::load_host_organism(Organism &organism) {
    x = organism.x;
    y = organism.y;
    vector_index = organism.vector_index;
    num_blocks = organism.anatomy._organism_blocks.size();

    gpuErrchk(cudaMalloc((SerializedOrganismBlockContainer**)&_organism_blocks,
                        sizeof(SerializedOrganismBlockContainer)*organism.anatomy._organism_blocks.size()))

    gpuErrchk(cudaMemcpy(_organism_blocks,
                         organism.anatomy._organism_blocks.data(),
                         sizeof(SerializedOrganismBlockContainer)*organism.anatomy._organism_blocks.size(),
                         cudaMemcpyHostToDevice))
}


__global__
void apply_organism_change(CudaTransaction * d_transaction, RecCudaOrganism * d_orgs, BaseGridBlock * rec_grid, int width) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->organism_change_size) { return;}

    auto & no = d_transaction->organism_change[i_pos];

//    printf("%d %d", d_transaction->organism_change_size, no.vector_index);
    auto & o = d_orgs[no.vector_index];

    apply_new_organism(&o, &no);

    for (int i = 0; i < o.num_blocks; i++) {
        auto & b = o._organism_blocks[i];
        auto & wb = rec_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        wb.type = b.type;
        wb.rotation = b.rotation;
    }
}

__global__
void apply_food_change(CudaTransaction *d_transaction, BaseGridBlock *d_rec_grid, int sim_width, float *d_food_grid) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->food_change_size) { return;}

    auto & chg = d_transaction->food_change[i_pos];
    auto & num = d_food_grid[chg.x + chg.y * sim_width];
    num += chg.num;

    //TODO
    d_rec_grid[chg.x + chg.y * sim_width].type = chg.num > 0.99 ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
}

__device__
void subtract_difference(int x, int y, SerializedOrganismBlockContainer * _organism_blocks, int num_blocks) {
    for (int i = 0; i < num_blocks; i++) {
        auto & block = _organism_blocks[i];
        block.relative_x -= x;
        block.relative_y -= y;
    }
}

__device__
CVector2 recenter_to_existing(SerializedOrganismBlockContainer * _organism_blocks, int num_blocks) {
    int block_pos_in_vec = 0;
    //the position of a block will definitely never be bigger than this.
    CVector2 abs_pos{INT32_MAX/4, INT32_MAX/4};

    //will find the closest cell to the center
    int i = 0;
    for (int _i = 0; _i < num_blocks; _i++) {
        auto & block = _organism_blocks[_i];
        if (std::abs(block.relative_x) + std::abs(block.relative_y) < abs_pos.x + abs_pos.y) {
            abs_pos = {std::abs(block.relative_x), std::abs(block.relative_y)};
            block_pos_in_vec = i;
        }
        i++;
    }

    CVector2 new_center_pos = {_organism_blocks[block_pos_in_vec].relative_x, _organism_blocks[block_pos_in_vec].relative_y};

    subtract_difference(new_center_pos.x, new_center_pos.y, _organism_blocks, num_blocks);
    return {new_center_pos.x, new_center_pos.y};
}

__device__
CVector2 recenter_to_imaginary(SerializedOrganismBlockContainer * _organism_blocks, int num_blocks) {
    CVector2 min{0, 0};
    CVector2 max{0, 0};

    for (int i = 0; i < num_blocks; i++) {
        auto & block = _organism_blocks[i];
        if (block.relative_x < min.x) {min.x = block.relative_x;}
        if (block.relative_y < min.y) {min.y = block.relative_y;}
        if (block.relative_x > max.x) {max.x = block.relative_x;}
        if (block.relative_y > max.y) {max.y = block.relative_y;}
    }

    auto diff_x = (max.x - min.x) / 2 + min.x;
    auto diff_y = (max.y - min.y) / 2 + min.y;

    subtract_difference(diff_x, diff_y, _organism_blocks, num_blocks);

    return {diff_x, diff_y};
}

__device__
CVector2 recenter_blocks(bool imaginary_center, SerializedOrganismBlockContainer * _organism_blocks, int num_blocks) {
    if (imaginary_center) {
        return recenter_to_imaginary(_organism_blocks, num_blocks);
    } else {
        return recenter_to_existing(_organism_blocks, num_blocks);
    }
}

__global__
void apply_recenter(CudaTransaction * d_transaction, RecCudaOrganism * d_orgs, int num_orgs, bool recenter) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= num_orgs) { return;}

    auto & organism = d_orgs[i_pos];

    auto vec = recenter_blocks(recenter, organism._organism_blocks, organism.num_blocks);

    auto temp = get_pos(organism.rotation, vec.x, vec.y);

    organism.x += temp.x;
    organism.y += temp.y;
}

__global__
void apply_dead_organisms(CudaTransaction *d_transaction, RecCudaOrganism *d_orgs, BaseGridBlock *d_rec_grid, int width,
                          float *d_food_grid) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->dead_organisms_size) { return;}

    auto & dc = d_transaction->dead_organisms[i_pos];
    auto & o = d_orgs[dc];
    for (int i = 0; i < o.num_blocks; i++) {
        auto & b = o._organism_blocks[i];

        auto & wb = d_rec_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        wb.type = BlockTypes::FoodBlock;

        //TODO
        auto & fb = d_food_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        fb += 1;
    }
}

__global__
void apply_move_change(CudaTransaction *d_transaction, BaseGridBlock *d_rec_grid, RecCudaOrganism *d_orgs, int width,
                       float *d_food_grid) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->move_change_size) { return;}

    auto & chg = d_transaction->move_change[i_pos];
    auto & o = d_orgs[chg.vector_index];

    for (int i = 0; i < o.num_blocks; i++) {
        auto & b = o._organism_blocks[i];
        auto & wb = d_rec_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        auto & fb = d_food_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        //TODO
        if (fb > 0.99) {
            wb.type = BlockTypes::FoodBlock;
        } else {
            wb.type = BlockTypes::EmptyBlock;
        }
    }

    o.rotation = chg.rotation;
    o.x = chg.x;
    o.y = chg.y;

    for (int i = 0; i < o.num_blocks; i++) {
        auto & b = o._organism_blocks[i];
        auto & wb = d_rec_grid[o.x + get_pos(o.rotation, b.relative_x, b.relative_y).x + (o.y + get_pos(o.rotation, b.relative_x, b.relative_y).y) * width];
        wb.type = b.type;
        wb.rotation = b.rotation;
    }
}

__global__
void apply_wall_change(CudaTransaction * d_transaction, BaseGridBlock * d_rec_grid, int sim_width) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->wall_change_size) { return;}

    auto & chg = d_transaction->wall_change[i_pos];
    d_rec_grid[chg.x + chg.y * sim_width].type = chg.added ? BlockTypes::WallBlock : BlockTypes::EmptyBlock;
}

__global__
void apply_compressed_change(CudaTransaction * d_transaction, RecCudaOrganism * d_orgs) {
    auto i_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pos >= d_transaction->compressed_change_size) { return;}

    auto & pair = d_transaction->compressed_change[i_pos];

    swap(&d_orgs[pair.first], &d_orgs[pair.second]);
}


void RecordingReconstructorCUDA::start_reconstruction(int width, int height) {
    grid_width = width;
    grid_height = height;

    std::vector<BaseGridBlock> temp(width*height, BaseGridBlock{BlockTypes::EmptyBlock});
    std::vector<float> temp2(grid_width*grid_height, 0);

    gpuErrchk(cudaMalloc((BaseGridBlock**)&d_rec_grid, sizeof(BaseGridBlock)*width*height))
    gpuErrchk(cudaMalloc((BaseGridBlock**)&d_food_grid, sizeof(BaseGridBlock)*width*height))

    gpuErrchk(cudaMemcpy(d_rec_grid,
                         temp.data(),
                         sizeof(BaseGridBlock)*temp.size(),
                         cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_food_grid,
                         temp2.data(),
                         sizeof(BaseGridBlock)*temp2.size(),
                         cudaMemcpyHostToDevice))
}

void RecordingReconstructorCUDA::apply_transaction(Transaction &transaction) {
    if (transaction.reset) {
        apply_reset(transaction);
    } else if (transaction.starting_point) {
        apply_starting_point(transaction);
    } else {
        apply_normal(transaction);
    }
}

void RecordingReconstructorCUDA::apply_starting_point(Transaction &transaction) {
    //clear grid
    std::vector<BaseGridBlock> temp(grid_width*grid_height, BaseGridBlock{BlockTypes::EmptyBlock});
    gpuErrchk(cudaMemcpy(d_rec_grid,
                         temp.data(),
                         sizeof(BaseGridBlock)*temp.size(),
                         cudaMemcpyHostToDevice))
    std::vector<float> temp2(grid_width*grid_height, 0);
    gpuErrchk(cudaMemcpy(d_food_grid,
                         temp2.data(),
                         sizeof(BaseGridBlock)*temp2.size(),
                         cudaMemcpyHostToDevice))

    for (int i = 0; i < num_orgs; i++) {
        cudaFree(d_rec_orgs[i]._organism_blocks);
    }
    cudaFree(d_rec_orgs);

    recenter_to_imaginary_position = transaction.recenter_to_imaginary_pos;

    apply_normal(transaction);
}

void RecordingReconstructorCUDA::apply_reset(Transaction & transaction) {
    std::vector<BaseGridBlock> temp(grid_width*grid_height, BaseGridBlock{BlockTypes::EmptyBlock});
    gpuErrchk(cudaMemcpy(d_rec_grid,
                         temp.data(),
                         sizeof(BaseGridBlock)*temp.size(),
                         cudaMemcpyHostToDevice))
    std::vector<float> temp2(grid_width*grid_height, 0);
    gpuErrchk(cudaMemcpy(d_food_grid,
                         temp2.data(),
                         sizeof(BaseGridBlock)*temp2.size(),
                         cudaMemcpyHostToDevice))

    for (int i = 0; i < num_orgs; i++) {
        cudaFree(d_rec_orgs[i]._organism_blocks);
    }
    cudaFree(d_rec_orgs);

    recenter_to_imaginary_position = transaction.recenter_to_imaginary_pos;

    transaction.dead_organisms.clear();

    apply_normal(transaction);
}

int block_size = 32;

void RecordingReconstructorCUDA::apply_normal(Transaction &transaction) {
    prepare_transaction(transaction);

    int num;

    //TODO

    if (d_transaction->wall_change_size != 0) {
    num = d_transaction->food_change_size / block_size + block_size;
        apply_food_change<<<num, block_size>>>(d_transaction, d_rec_grid, grid_width, d_food_grid);
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (d_transaction->wall_change_size != 0) {
    if (recenter_to_imaginary_position != transaction.recenter_to_imaginary_pos) { recenter_to_imaginary_position = transaction.recenter_to_imaginary_pos;
    num = num_orgs / block_size + block_size;
    apply_recenter<<<num, block_size>>>(d_transaction, d_rec_orgs, num_orgs, recenter_to_imaginary_position);}
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (d_transaction->wall_change_size != 0) {
    num = d_transaction->dead_organisms_size / block_size + block_size;
        apply_dead_organisms<<<num, block_size>>>(d_transaction, d_rec_orgs, d_rec_grid, grid_width, d_food_grid);
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (d_transaction->wall_change_size != 0) {
    num = d_transaction->compressed_change_size / block_size + block_size;
    apply_compressed_change<<<num, block_size>>>(d_transaction, d_rec_orgs);
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (d_transaction->wall_change_size != 0) {
    num = d_transaction->move_change_size / block_size + block_size;
        apply_move_change<<<num, block_size>>>(d_transaction, d_rec_grid, d_rec_orgs, grid_width, d_food_grid);
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (d_transaction->wall_change_size != 0) {
    num = d_transaction->wall_change_size / block_size + block_size;
    apply_wall_change<<<num, block_size>>>(d_transaction, d_rec_grid, grid_width);
    gpuErrchk(cudaDeviceSynchronize())
    }

    if (!transaction.organism_change.empty()) {
        int  last_vector_pos = std::max_element(transaction.organism_change.begin(), transaction.organism_change.end(),
                                           [](Organism &o1, Organism &o2) {
                                                return o1.vector_index < o2.vector_index;
                                            })->vector_index + 1;
        if (last_vector_pos > num_orgs) {
            RecCudaOrganism *temp;
            //TODO
            cudaMallocManaged((RecCudaOrganism **) &temp, sizeof(RecCudaOrganism) * last_vector_pos);

            for (int i = 0; i < num_orgs; i++) {
                temp[i] = d_rec_orgs[i];
            }

            num_orgs = last_vector_pos;
            cudaFree(d_rec_orgs);
            d_rec_orgs = temp;
        }
        num = d_transaction->organism_change_size / block_size + block_size;
        apply_organism_change<<<num, block_size>>>(d_transaction, d_rec_orgs, d_rec_grid, grid_width);
        gpuErrchk(cudaDeviceSynchronize())
    }
}

//TODO ok i need two different sizes, one for actual malloced side and the other is for kernel to know the size of
// transaction data
void RecordingReconstructorCUDA::prepare_transaction(Transaction &transaction) {
    if (d_transaction == nullptr) {
        gpuErrchk(cudaMallocManaged((CudaTransaction**)&d_transaction, sizeof(CudaTransaction)))
        *d_transaction = CudaTransaction{};
    }

    if (transaction.organism_change.size() > d_transaction->organism_change_size_length) {
        cudaFree(d_transaction->organism_change);
        gpuErrchk(cudaMalloc((RecCudaOrganism**)&d_transaction->organism_change, sizeof(RecCudaOrganism)*transaction.organism_change.size()))
        d_transaction->organism_change_size_length = transaction.organism_change.size();
    }
    d_transaction->organism_change_size = transaction.organism_change.size();
    std::vector<RecCudaOrganism> temp(transaction.organism_change.size());
    for (int i = 0; i < temp.size(); i++) {
        temp[i].load_host_organism(transaction.organism_change[i]);
    }

    gpuErrchk(cudaMemcpy(d_transaction->organism_change,
                         temp.data(),
                         sizeof(RecCudaOrganism)*temp.size(),
                         cudaMemcpyHostToDevice))


    if (transaction.food_change.size() > d_transaction->food_change_size_length) {
        cudaFree(d_transaction->food_change);
        gpuErrchk(cudaMalloc((FoodChange**)&d_transaction->food_change, sizeof(FoodChange)*transaction.food_change.size()))
        d_transaction->food_change_size_length = transaction.food_change.size();
    }
    d_transaction->food_change_size = transaction.food_change.size();
    gpuErrchk(cudaMemcpy(d_transaction->food_change,
                         transaction.food_change.data(),
                         sizeof(FoodChange)*transaction.food_change.size(),
                         cudaMemcpyHostToDevice))

    if (transaction.dead_organisms.size() > d_transaction->dead_organisms_size_length) {
        cudaFree(d_transaction->dead_organisms);
        gpuErrchk(cudaMalloc((int**)&d_transaction->dead_organisms, sizeof(int)*transaction.dead_organisms.size()))
        d_transaction->dead_organisms_size_length = transaction.dead_organisms.size();
    }
    d_transaction->dead_organisms_size = transaction.dead_organisms.size();
    gpuErrchk(cudaMemcpy(d_transaction->dead_organisms,
                         transaction.dead_organisms.data(),
                         sizeof(int)*transaction.dead_organisms.size(),
                         cudaMemcpyHostToDevice))

    if (transaction.move_change.size() > d_transaction->move_change_size_length) {
        cudaFree(d_transaction->move_change);
        gpuErrchk(cudaMalloc((MoveChange**)&d_transaction->move_change, sizeof(MoveChange)*transaction.move_change.size()))
        d_transaction->move_change_size_length = transaction.move_change.size();
    }
    d_transaction->move_change_size = transaction.move_change.size();
    gpuErrchk(cudaMemcpy(d_transaction->move_change,
                         transaction.move_change.data(),
                         sizeof(MoveChange)*transaction.move_change.size(),
                         cudaMemcpyHostToDevice))

    if (transaction.user_wall_change.size() > d_transaction->wall_change_size_length) {
        cudaFree(d_transaction->wall_change);
        gpuErrchk(cudaMalloc((WallChange**)&d_transaction->wall_change, sizeof(WallChange)*transaction.user_wall_change.size()))
        d_transaction->wall_change_size_length = transaction.user_wall_change.size();
    }
    d_transaction->wall_change_size = transaction.user_wall_change.size();
    gpuErrchk(cudaMemcpy(d_transaction->wall_change,
                         transaction.user_wall_change.data(),
                         sizeof(WallChange)*transaction.user_wall_change.size(),
                         cudaMemcpyHostToDevice))

    if (transaction.compressed_change.size() > d_transaction->compressed_change_size_length) {
        cudaFree(d_transaction->compressed_change);
        gpuErrchk(cudaMalloc((std::pair<int, int>**)&d_transaction->compressed_change, sizeof(std::pair<int, int>)*transaction.compressed_change.size()))
        d_transaction->compressed_change_size_length = transaction.compressed_change.size();
    }
    d_transaction->compressed_change_size = transaction.compressed_change.size();
    gpuErrchk(cudaMemcpy(d_transaction->compressed_change,
                         transaction.compressed_change.data(),
                         sizeof(std::pair<int, int>)*transaction.compressed_change.size(),
                         cudaMemcpyHostToDevice))
}

void RecordingReconstructorCUDA::prepare_image_creation(int image_width, int image_height, std::vector<int> lin_width,
                                                        std::vector<int> lin_height,
                                                        std::vector<Vector2<int>> width_img_boundaries,
                                                        std::vector<Vector2<int>> height_img_boundaries,
                                                        TexturesContainer &container,
                                                        ColorContainer *color_container) {
    this->image_width = image_width;
    this->image_height = image_height;
    this->color_container = color_container;

    std::vector<Differences> d{};

    creator.check_if_changed(image_width, image_height, grid_width, grid_height, width_img_boundaries.size(),
                             height_img_boundaries.size(), lin_width.size(), lin_height.size(), 0, true);
    creator.copy_to_device(lin_width, lin_height, width_img_boundaries, height_img_boundaries, d);
    creator.copy_textures(container);
}

void RecordingReconstructorCUDA::make_image(std::vector<unsigned char> &image_vector) {
    //TODO i think that I need to load symbols every time. idk though.
    creator.load_symbols(color_container);
    CUDAImageCreator::cuda_call_create_image(image_width, image_height, image_vector, 32, grid_width, grid_height,
                                             creator.d_lin_width, creator.d_lin_height, creator.d_width_img_boundaries,
                                             creator.d_height_img_boundaries, creator.d_image_vector, d_rec_grid,
                                             creator.d_textures, height_img_boundaries_size, width_img_boundaries_size,
                                             true);
}

void RecordingReconstructorCUDA::finish_image_creation() {
    creator.free();
    creator.free_textures();
}

void RecordingReconstructorCUDA::finish_reconstruction() {
    cudaFree(d_rec_grid);

    for (int i = 0; i < num_orgs; i++) {
        auto & org = d_rec_orgs;
        cudaFree(org->_organism_blocks);
    }
    cudaFree(d_rec_orgs);
    cudaFree(d_food_grid);
    cudaFree(d_transaction);

    num_orgs = 0;
    grid_width = 0;
    grid_height = 0;
    image_height = 0;
    image_width = 0;
    width_img_boundaries_size = 0;
    height_img_boundaries_size = 0;
}
