//
// Created by SpaceEye on 12.06.22.
//

#ifndef THELIFEENGINECPP_OBSERVATIONSTUFF_H
#define THELIFEENGINECPP_OBSERVATIONSTUFF_H

#include <cstdint>

#include "../../BlockTypes.hpp"
#include "Rotation.h"

struct Observation {
    BlockTypes type = EmptyBlock;
    int32_t distance = 0;
    //local rotation
    Rotation eye_rotation = Rotation::UP;
};

#endif //THELIFEENGINECPP_OBSERVATIONSTUFF_H
