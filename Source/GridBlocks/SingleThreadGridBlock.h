// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 28.05.2022.
//

#ifndef THELIFEENGINECPP_ATOMICGRIDBLOCK_H
#define THELIFEENGINECPP_ATOMICGRIDBLOCK_H

struct SingleThreadGridBlock {
    BlockTypes type = BlockTypes::EmptyBlock;
    Rotation rotation = Rotation::UP;

    int32_t organism_index = -1;
    SingleThreadGridBlock()=default;
    SingleThreadGridBlock(const SingleThreadGridBlock & block)=default;
    explicit SingleThreadGridBlock(BlockTypes type, Rotation rotation = Rotation::UP,
                                   int32_t organism_index = -1):
            type(type), rotation(rotation), organism_index(organism_index) {}
    SingleThreadGridBlock& operator=(const SingleThreadGridBlock & block) = default;;
};


#endif //THELIFEENGINECPP_ATOMICGRIDBLOCK_H
