//
// Created by spaceeye on 01.04.2022.
//

#ifndef THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
#define THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H

struct BlockParameters {
    float food_cost_modifier = 1; float chance_weight = 1;
};

struct OrganismBlockParameters {
    BlockParameters MouthBlock    = {1, 1};
    BlockParameters ProducerBlock = {1, 1};
    BlockParameters MoverBlock    = {1, 1};
    BlockParameters KillerBlock   = {1, 1};
    BlockParameters ArmorBlock    = {1, 1};
    BlockParameters EyeBlock      = {1, 1};
};


#endif //THELIFEENGINECPP_ORGANISMBLOCKPARAMETERS_H
