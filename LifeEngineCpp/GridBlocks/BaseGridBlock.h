//#include <chrono>
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_BASEGRIDBLOCK_H
#define THELIFEENGINECPP_BASEGRIDBLOCK_H

#include <iostream>
#include "../BlockTypes.hpp"
#include "../Organism/CPU/Rotation.h"

struct Neighbors {
    //has neighbor
    bool up    = false;
    bool left  = false;
    bool down  = false;
    bool right = false;
};

class Organism;

struct BaseGridBlock{
public:
    BlockTypes type = BlockTypes::EmptyBlock;
    Rotation rotation = Rotation::UP;
    Neighbors neighbors{};

    Organism * organism = nullptr;
    BaseGridBlock()=default;
    BaseGridBlock(const BaseGridBlock & block): type(block.type),
                                                rotation(block.rotation),
                                                neighbors(block.neighbors),
                                                organism(block.organism) {}
    explicit BaseGridBlock(BlockTypes type, Rotation rotation = Rotation::UP, Neighbors neighbors = Neighbors{},
                           Organism * organism = nullptr):
                            type(type), rotation(rotation), neighbors(neighbors), organism(organism) {}
    BaseGridBlock& operator=(const BaseGridBlock & block) = default;
};
#endif //THELIFEENGINECPP_BASEGRIDBLOCK_H
