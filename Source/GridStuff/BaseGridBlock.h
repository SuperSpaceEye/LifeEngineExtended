// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_BASEGRIDBLOCK_H
#define THELIFEENGINECPP_BASEGRIDBLOCK_H

#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/enums/Rotation.h"

struct BaseGridBlock{
public:
    BlockTypes type = BlockTypes::EmptyBlock;
    Rotation rotation = Rotation::UP;

    BaseGridBlock()=default;
    BaseGridBlock(const BaseGridBlock & block) = default;
    explicit BaseGridBlock(BlockTypes type, Rotation rotation = Rotation::UP): type(type),
                                                                               rotation(rotation) {}
    BaseGridBlock& operator=(const BaseGridBlock & block) = default;
};
#endif //THELIFEENGINECPP_BASEGRIDBLOCK_H
