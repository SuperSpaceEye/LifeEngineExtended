//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISMBLOCK_H
#define THELIFEENGINECPP_ORGANISMBLOCK_H

#include "../../GridBlocks/BaseGridBlock.h"

struct OrganismBlock: public BaseGridBlock {
public:
    OrganismBlock();
    explicit OrganismBlock(BlockTypes type) : BaseGridBlock(type) {}
};


#endif //THELIFEENGINECPP_ORGANISMBLOCK_H
