//
// Created by spaceeye on 16.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCLOGICCONTAINER_H
#define LIFEENGINEEXTENDED_OCCLOGICCONTAINER_H

#include "../../GridStuff/BaseGridBlock.h"

struct OCCBlock: public BaseGridBlock {
    int parent_block_pos = -1;
    uint64_t counter = 0;
    OCCBlock()=default;
};

struct OCCSpace {
    int parent_block_pos = -1;
    uint64_t counter = 0;
};

struct OCCSerializedProducingSpace {
    int producer = 0;
    int x = 0;
    int y = 0;
    OCCSerializedProducingSpace(int producer, int x, int y): producer(producer), x(x), y(y){};
};

struct OCCLogicContainer {
    int occ_width  = 0;
    int occ_height = 0;
    uint64_t main_counter = 0;
    uint64_t spaces_counter = 0;

    std::vector<OCCBlock> occ_main_block_construction_space;
    //for producing/eating/killing spaces
    std::vector<OCCSpace> occ_producing_space;
    std::vector<OCCSpace> occ_eating_space;
    std::vector<OCCSpace> occ_killing_space;
};

#endif //LIFEENGINEEXTENDED_OCCLOGICCONTAINER_H
