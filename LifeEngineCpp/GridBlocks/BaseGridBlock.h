//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_BASEGRIDBLOCK_H
#define THELIFEENGINECPP_BASEGRIDBLOCK_H

#include <iostream>
#include "../BlockTypes.hpp"
#include "../Organism/Rotation.h"
//#include "../Organism/Organism.h"

struct Neighbors {
    //has neighbor
    bool up    = false;
    bool left  = false;
    bool down  = false;
    bool right = false;
};

class Organism;

class BaseGridBlock {
public:
    BlockTypes type = BlockTypes::EmptyBlock;
    Rotation rotation = Rotation::UP;
    Neighbors neighbors{};

    Organism * organism = nullptr;
    BaseGridBlock()=default;
    explicit BaseGridBlock(BlockTypes type, Rotation rotation = Rotation::UP, Neighbors neighbors = Neighbors{},
                           Organism * organism = nullptr):
                            type(type), rotation(rotation), neighbors(neighbors), organism(organism) {}
};
#endif //THELIFEENGINECPP_BASEGRIDBLOCK_H
