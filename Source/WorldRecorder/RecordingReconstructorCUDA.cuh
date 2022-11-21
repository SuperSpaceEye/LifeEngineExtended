//
// Created by spaceeye on 25.10.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTORCUDA_CUH
#define LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTORCUDA_CUH

#include "../Stuff/cuda_image_creator.cuh"
#include "WorldRecorder.h"

class RecCudaOrganism {
public:
    int x = -1;
    int y = -1;
    int vector_index = -1;
    Rotation rotation;

    int num_blocks = -1;
    SerializedOrganismBlockContainer * _organism_blocks = nullptr;

    void load_host_organism(Organism & organism);
};

struct CudaTransaction {
    RecCudaOrganism * organism_change = nullptr;
    FoodChange * food_change = nullptr;
    int * dead_organisms = nullptr;
    MoveChange * move_change = nullptr;
    WallChange * wall_change = nullptr;
    std::pair<int, int> * compressed_change = nullptr;

    //length of a current transaction
    int organism_change_size = 0;
    int food_change_size = 0;
    int dead_organisms_size = 0;
    int move_change_size = 0;
    int wall_change_size = 0;
    int compressed_change_size = 0;

    //length of a container
    int organism_change_size_length = 0;
    int food_change_size_length = 0;
    int dead_organisms_size_length = 0;
    int move_change_size_length = 0;
    int wall_change_size_length = 0;
    int compressed_change_size_length = 0;

    bool starting_point;
    bool recenter_to_imaginary_pos;
    bool reset = false;
    bool uses_occ = false;
};

class RecordingReconstructorCUDA {
    int image_width;
    int image_height;
    int width_img_boundaries_size;
    int height_img_boundaries_size;

    CUDAImageCreator creator{};

    int *d_lin_width = nullptr;
    int *d_lin_height = nullptr;
    Vector2<int> *d_width_img_boundaries = nullptr;
    Vector2<int> *d_height_img_boundaries = nullptr;
    //TODO most of the time is wasted copying result image from device. Possible optimisation may be to allocate
    // image vector as image_width*image_height*buffer_size. The device will write to this buffer as
    // x + y * width + buffer_pos*width*height. Then, when the buffer is full, just copy the whole memory block, instead
    // of making many copy memory calls
    unsigned char *d_image_vector = nullptr;
    BaseGridBlock *d_rec_grid = nullptr;
    CudaTextureHolder *d_textures = nullptr;
    ColorContainer * color_container = nullptr;
    float *d_food_grid = nullptr;

    int num_orgs;
    // Using cudaMallocManaged is probably efficient enough as reconstruction itself takes minuscule amount of execution
    // time.
    RecCudaOrganism * d_rec_orgs = nullptr;
    CudaTransaction * d_transaction = nullptr;

    int grid_width;
    int grid_height;
    bool recenter_to_imaginary_position;

    void apply_starting_point(Transaction & transaction);
    void apply_reset(Transaction & transaction);
    void apply_normal(Transaction & transaction);

    void prepare_transaction(Transaction & transaction);
public:
    RecordingReconstructorCUDA()=default;

    void start_reconstruction(int width, int height);
    void prepare_image_creation(int image_width, int image_height, std::vector<int> lin_width,
                                std::vector<int> lin_height,
                                std::vector<Vector2<int>> width_img_boundaries,
                                std::vector<Vector2<int>> height_img_boundaries,
                                TexturesContainer &container,
                                ColorContainer *color_container);
    void apply_transaction(Transaction & transaction);
    void make_image(std::vector<unsigned char> & image_vector);
    void finish_reconstruction();
    void finish_image_creation();
};


#endif //LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTORCUDA_CUH
