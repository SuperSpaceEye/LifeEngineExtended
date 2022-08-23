// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 01.04.2022.
//

#ifndef THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
#define THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H

struct BParameters {
    // food_cost_modifier - how much food does organism_index have to spend on one block when creating a child
    // life_point_amount - how much organism_index gains life points from this block
    float food_cost_modifier = 1; float life_point_amount = 1; float lifetime_weight = 1; float chance_weight = 1;
};

struct OrganismBlockParameters {
    BParameters MouthBlock    = {1, 1, 1, 1};
    BParameters ProducerBlock = {1, 1, 1, 1};
    BParameters MoverBlock    = {1, 1, 1, 1};
    BParameters KillerBlock   = {1, 1, 1, 1};
    BParameters ArmorBlock    = {1, 1, 1, 1};
    BParameters EyeBlock      = {1, 1, 1, 1};
};

enum class BlocksNames {
    MouthBlock,
    ProducerBlock,
    MoverBlock,
    KillerBlock,
    ArmorBlock,
    EyeBlock
};

enum class ParametersNames {
    FoodCostModifier,
    LifePointAmount,
    LifetimeWeight,
    ChanceWeight,
};

#endif //THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
