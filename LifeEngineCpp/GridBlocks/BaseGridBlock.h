//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_BASEGRIDBLOCK_H
#define THELIFEENGINECPP_BASEGRIDBLOCK_H

#include <iostream>
#include "../Stuff/BlockTypes.hpp"
#include "../Organism/CPU/Rotation.h"

class Organism;

struct BaseGridBlock{
public:
    BlockTypes type = BlockTypes::EmptyBlock;
    Rotation rotation = Rotation::UP;

    Organism * organism = nullptr;
    BaseGridBlock()=default;
    BaseGridBlock(const BaseGridBlock & block): type(block.type),
                                                rotation(block.rotation),
                                                organism(block.organism) {}
    explicit BaseGridBlock(BlockTypes type, Rotation rotation = Rotation::UP,
                           Organism * organism = nullptr):
                            type(type), rotation(rotation), organism(organism) {}
    BaseGridBlock& operator=(const BaseGridBlock & block) = default;
};
#endif //THELIFEENGINECPP_BASEGRIDBLOCK_H
