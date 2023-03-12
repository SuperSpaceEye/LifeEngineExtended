// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 01.04.2022.
//

#ifndef THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
#define THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H

#include "Stuff/enums/BlockTypes.hpp"

struct BParameters {
    // food_cost_modifier - how much food does organism have to spend on one block when creating a child
    // life_point_amount - how much organism gains life points from this block
    float food_cost = 1; float life_point_amount = 1; float lifetime_weight = 1; float chance_weight = 1;
};

struct OrganismBlockParameters {
    //parameter array
    std::array<BParameters, NUM_ORGANISM_BLOCKS> pa;
};

enum class ParametersNames {
    FoodCostModifier,
    LifePointAmount,
    LifetimeWeight,
    ChanceWeight,
};

#endif //THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
