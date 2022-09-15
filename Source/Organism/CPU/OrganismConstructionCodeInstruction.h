//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H

#include "Anatomy.h"

enum class OCCInstruction {
    ShiftUp,
    ShiftLeft,
    ShiftDown,
    ShiftRight,

    ApplyRotationUp,
    ApplyRotationLeft,
    ApplyRotationDown,
    ApplyRotationRight,

    SetRotationUp,
    SetRotationLeft,
    SetRotationDown,
    SetRotationRight,

    ResetToOrigin,
    SetOrigin,

    SetBlockMouth,
    SetBlockProducer,
    SetBlockMover,
    SetBlockKiller,
    SetBlockArmor,
    SetBlockEye
};
#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
