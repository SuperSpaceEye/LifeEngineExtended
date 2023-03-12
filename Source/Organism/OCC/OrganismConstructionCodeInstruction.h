//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H

#include "Organism/Anatomy/Anatomy.h"

//18
//6 + 6
//30 total

const int NON_SET_BLOCK_OCC_INSTRUCTIONS = 18;
const int SET_BLOCK_OCC_INSTRUCTIONS = 12;

enum class OCCInstruction {
    ShiftUp,
    ShiftUpLeft,
    ShiftLeft,
    ShiftLeftDown,
    ShiftDown,
    ShiftDownRight,
    ShiftRight,
    ShiftRightUp,

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
    SetBlockEye,

    SetUnderBlockMouth,
    SetUnderBlockProducer,
    SetUnderBlockMover,
    SetUnderBlockKiller,
    SetUnderBlockArmor,
    SetUnderBlockEye
};

const std::array<std::string, 30> OCC_INSTRUCTIONS_NAME {
        "Shift Up",
        "Shift Up Left",
        "Shift Left",
        "Shift Left Down",
        "Shift Down",
        "Shift Down Right",
        "Shift Right",
        "Shift Right Up",

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
        "Set Block Eye",

        "Set Under Block Mouth",
        "Set Under Block Producer",
        "Set Under Block Mover",
        "Set Under Block Killer",
        "Set Under Block Armor",
        "Set Under Block Eye"
};

const std::array<std::string, 30> OCC_INSTRUCTIONS {
        "ShiftUp",
        "ShiftUpLeft",
        "ShiftLeft",
        "ShiftLeftDown",
        "ShiftDown",
        "ShiftDownRight",
        "ShiftRight",
        "ShiftRightUp",

        "ApplyRotationUp",
        "ApplyRotationLeft",
        "ApplyRotationDown",
        "ApplyRotationRight",

        "SetRotationUp",
        "SetRotationLeft",
        "SetRotationDown",
        "SetRotationRight",

        "ResetToOrigin",
        "SetOrigin",

        "SetBlockMouth",
        "SetBlockProducer",
        "SetBlockMover",
        "SetBlockKiller",
        "SetBlockArmor",
        "SetBlockEye",

        "SetUnderBlockMouth",
        "SetUnderBlockProducer",
        "SetUnderBlockMover",
        "SetUnderBlockKiller",
        "SetUnderBlockArmor",
        "SetUnderBlockEye"
};

const std::array<std::string, 30> OCC_INSTRUCTIONS_SHORT {
        "SU",
        "SUL",
        "SL",
        "SLD",
        "SD",
        "SDR",
        "SR",
        "SRU",

        "ARU",
        "ARL",
        "ARD",
        "ARR",

        "SRTU",
        "SRL",
        "SRD",
        "SRR",

        "RTO",
        "SO",

        "SBM",
        "SBP",
        "SBMV",
        "SBK",
        "SBA",
        "SBE",

        "SUBM",
        "SUBP",
        "SUBMV",
        "SURK",
        "SUBA",
        "SUBE"
};

// TODO is this faster than using hashmap? idk
int inline get_index_of_occ_instruction_name(std::string &name) {
    for (int i = 0; i < OCC_INSTRUCTIONS_NAME.size(); i++) {
        if (name == OCC_INSTRUCTIONS_NAME[i]) {
            return i;
        }
    }
    return -1;
}

int inline get_index_of_occ_instruction(std::string &name) {
    for (int i = 0; i < OCC_INSTRUCTIONS.size(); i++) {
        if (name == OCC_INSTRUCTIONS[i]) {
            return i;
        }
    }
    return -1;
}

int inline get_index_of_occ_instruction_short(std::string &name) {
    for (int i = 0; i < OCC_INSTRUCTIONS_SHORT.size(); i++) {
        if (name == OCC_INSTRUCTIONS_SHORT[i]) {
            return i;
        }
    }
    return -1;
}
#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEINSTRUCTION_H
