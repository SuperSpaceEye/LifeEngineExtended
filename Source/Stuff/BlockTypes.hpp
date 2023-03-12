// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 23.03.2022.
//

/*
 * Adding new block type. The order of types is important !!!
 * 1) Add new type below last organism block type and add appropriate string names
 * 2) go to Source/Organism/CPU/OrganismConstructionCodeInstruction.h
 * 3) add new set instructions for new type and update SET_BLOCK_OCC_INSTRUCTIONS
 * short instructions shouldn't have the same name
 * if block has space
 *      4) go to Source/Organism/CPU/AnatomyContainers.h and modify SerializedOrganismStructureContainer
 *      5) go to Source/Organism/CPU/SimpleAnatomyMutationLogic.h
 *      6) go to Source/Organism/CPU/Anatomy.cpp make_container
 *      7) go to Source/Stuff/DataSavingFunctions.cpp {read/write}_organism_anatomy
 *      8) go to Source/Organism/CPU/OrganismConstructionCode.cpp
 * 9) go to Source/Stuff/textures.h and add new default texture.
 */

#ifndef THELIFEENGINECPP_BLOCKTYPES_HPP
#define THELIFEENGINECPP_BLOCKTYPES_HPP

#include <array>
#include <string>
#include <string_view>

enum class BlockTypes {
    EmptyBlock,
    MouthBlock,
    ProducerBlock,
    MoverBlock,
    KillerBlock,
    ArmorBlock,
    EyeBlock,

    FoodBlock,
    WallBlock,
};

const std::array<std::string, 9> BLOCK_NAMES {"Empty", "Mouth", "Producer", "Mover", "Killer", "Armor", "Eye", "Food", "Wall"};
const std::array<std::string, 6> ORGANISM_BLOCK_NAMES {"Mouth", "Producer", "Mover", "Killer", "Armor", "Eye"};
constexpr std::string_view SW_ORGANISM_BLOCK_NAMES[6] {"mouth", "producer", "mover", "killer", "armor", "eye"};
constexpr int NUM_ORGANISM_BLOCKS = ORGANISM_BLOCK_NAMES.size();
constexpr int NUM_WORLD_BLOCKS = BLOCK_NAMES.size();

#endif //THELIFEENGINECPP_BLOCKTYPES_HPP
