//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_BASEGRIDBLOCK_H
#define THELIFEENGINECPP_BASEGRIDBLOCK_H

#include "../BlockTypes.h"

class BaseGridBlock {
public:
    BlockTypes type = BlockTypes::EmptyBlock;
    BaseGridBlock()=default;
};


#endif //THELIFEENGINECPP_BASEGRIDBLOCK_H
