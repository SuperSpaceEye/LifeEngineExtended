//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H

#include "Anatomy.h"

//18
//6
//24 total
enum class OCCInstruction {
    ShiftUp,
    ShiftUpLeft,
    ShiftLeft,
    ShiftLeftDown,
    ShiftDown,
    ShiftDownRight,
    ShiftRight,
    ShiftUpRight,

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

const std::array<std::string, 24> OCC_INSTRUCTIONS_NAME {
        "Shift Up",
        "Shift Up Left",
        "Shift Left",
        "Shift Left Down",
        "Shift Down",
        "Shift Down Right",
        "Shift Right",
        "Shift Up Right",

        "Apply Rotation Up",
        "Apply Rotation Left",
        "Apply Rotation Down",
        "Apply Rotation Right",

        "Set Rotation Up",
        "Set Rotation Left",
        "Set Rotation Down",
        "Set Rotation Right",

        "Reset To Origin",
        "Set Origin",

        "Set Block Mouth",
        "Set Block Producer",
        "Set Block Mover",
        "Set Block Killer",
        "Set Block Armor",
        "Set Block Eye"
};
#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
