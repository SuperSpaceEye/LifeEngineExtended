// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 23.03.2022.
//

#ifndef THELIFEENGINECPP_BLOCKTYPES_HPP
#define THELIFEENGINECPP_BLOCKTYPES_HPP

#include <vector>
#include <string>

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

const std::vector<std::string> blocks {"Empty", "Mouth", "Producer", "Mover", "Killer", "Armor", "Eye", "Food", "Wall"};
const std::vector<std::string> organism_blocks {"Mouth", "Producer", "Mover", "Killer", "Armor", "Eye"};

/*
 * Adding new block type.
 * 1) Add new type below EyeBlock
 * 2) go to Source/Organism/CPU/OrganismConstructionCodeInstruction.h
 * 3) add new set instructions for new type and update SET_BLOCK_OCC_INSTRUCTIONS
 * short instructions shouldn't have the same name
 * 4) go to Source/Organism/CPU/OrganismConstructionCode.cpp and modify
 * 5) go to Source/Organism/CPU/Anatomy.h and modify SerializedOrganismStructureContainer
 * (add counter for block type / add space for type)
 * 6) go to Anatomy class and also add counter and space
 * 7) go to Source/Organism/CPU/LegacyAnatomyMutationLogic and add stuff
 * 8) go to Source/Organism/CPU/Anatomy.cpp and add logic for new type
 * 9) go to Source/Stuff/textures.h and add new default texture. The order is important
 * 10) go to Source/Stuff/DataSavingFunctions.cpp {read/write}_organism_anatomy
 * 11) go to Source/UiFiles/statistics.ui and add new labels for type, transpile ui to h
 * 12) go to Source/UIWindows/MainWindow/MainWindow.cpp update_statistic_window
 * 13) go to Source/Containers/CPU/OrganismInfoContainer.h
 * 14) go to Source/UIWindows/OrganismEditor/OrganismEditor.h
 * 15) Source/UIWindows/OrganismEditor/OrganismEditor.cpp
 * 16) Source/Containers/CPU/OrganismBlockParameters.h
 * 17) Source/UIWindows/WorldEvents/ParametersList.cpp
 * 18) Source/UIWindows/MainWindow/MainWindowDataSavingStuff.cpp update_table_values
 * 19) World recorder
 * 20) Organism
 *
 *
 */

#endif //THELIFEENGINECPP_BLOCKTYPES_HPP
