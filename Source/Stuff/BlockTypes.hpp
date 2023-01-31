// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 23.03.2022.
//

#ifndef THELIFEENGINECPP_BLOCKTYPES_HPP
#define THELIFEENGINECPP_BLOCKTYPES_HPP

#include <array>
#include <string>
#include <string_view>

#include "ConstMap.h"

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

constexpr auto get_map(){
    return ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES>{};
};

//set map
constexpr void set_m(ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES>&m1,
                     const ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES>&m2) {
    for (auto & fi: SW_ORGANISM_BLOCK_NAMES) {
        m1[fi] = m2[fi];
    }
}

//get map parameter
constexpr int& get_mp(ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES>&m, BlockTypes type) {
    return m[SW_ORGANISM_BLOCK_NAMES[int(type) - 1]];
}

/*
 * Adding new block type.
 * 1) Add new type below EyeBlock
 * 2) go to Source/Organism/CPU/OrganismConstructionCodeInstruction.h
 * 3) add new set instructions for new type and update SET_BLOCK_OCC_INSTRUCTIONS
 * short instructions shouldn't have the same name
 * 4) go to Source/Organism/CPU/OrganismConstructionCode.cpp and modify
 * if block has space
 *      5) go to Source/Organism/CPU/AnatomyContainers.h and modify SerializedOrganismStructureContainer
 *      7) go to Source/Organism/CPU/SimpleAnatomyMutationLogic.h
 *      8) go to Source/Organism/CPU/Anatomy.cpp make_container
 *      10) go to Source/Stuff/DataSavingFunctions.cpp {read/write}_organism_anatomy
 * 9) go to Source/Stuff/textures.h and add new default texture. The order is important
 *
 * 11) go to Source/UiFiles/statistics.ui and add new labels for type, transpile ui to h
 * 12) go to Source/UIWindows/MainWindow/MainWindow.cpp update_statistic_window
 * 13) go to Source/Containers/CPU/OrganismInfoContainer.h
 * 19) World recorder
 *
 */

#endif //THELIFEENGINECPP_BLOCKTYPES_HPP
