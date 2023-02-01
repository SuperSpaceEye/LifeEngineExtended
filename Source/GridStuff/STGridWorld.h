// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 28.05.2022.
//

#ifndef THELIFEENGINECPP_ATOMICGRIDBLOCK_H
#define THELIFEENGINECPP_ATOMICGRIDBLOCK_H

#include <vector>

#include "../Stuff/BlockTypes.hpp"
#include "../Organism/CPU/Rotation.h"

//Singe Thread
struct STGridWorld {
private:
    int width = 0;
    int height = 0;
public:
    std::vector<BlockTypes> type_vec;
    std::vector<Rotation> rotation_vec;
    std::vector<float> food_vec;
    std::vector<int32_t> organism_index;

    void resize(int width, int height) {
        type_vec.resize(width*height, BlockTypes::EmptyBlock);
        rotation_vec.resize(width*height, Rotation::UP);
        food_vec.resize(width*height, 0);
        organism_index.resize(width*height, -1);
        this->width = width;
        this->height = height;
    }

    inline BlockTypes & get_type(int x, int y)        {return type_vec[x + y * width];}
    inline Rotation & get_rotation(int x, int y)      {return rotation_vec[x + y * width];}
    inline float & get_food_num(int x, int y)         {return food_vec[x + y * width];}
    inline int32_t & get_organism_index(int x, int y) {return organism_index[x + y * width];}
    inline bool add_food_num(int x, int y, float num, float max_food_num) {
        auto & fnum = food_vec[x + y * width];
        if (fnum + num > max_food_num) {
            return false;
        } else {
            if (fnum + num < 0) {return false;}
            fnum += num;
            return true;
        }
    }

    void clear() {
        type_vec = std::vector<BlockTypes>();
        rotation_vec = std::vector<Rotation>();
        food_vec = std::vector<float>();
        organism_index = std::vector<int32_t>();
    }
};

#endif //THELIFEENGINECPP_ATOMICGRIDBLOCK_H
