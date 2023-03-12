// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by SpaceEye on 12.06.22.
//

#ifndef THELIFEENGINECPP_OBSERVATIONSTUFF_H
#define THELIFEENGINECPP_OBSERVATIONSTUFF_H

#include <cstdint>

#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/enums/Rotation.h"

struct Observation {
    BlockTypes type = BlockTypes::EmptyBlock;
    int32_t distance = 0;
    //local rotation
    Rotation eye_rotation = Rotation::UP;
};

#endif //THELIFEENGINECPP_OBSERVATIONSTUFF_H
