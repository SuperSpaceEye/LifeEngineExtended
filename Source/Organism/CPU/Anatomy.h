// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ANATOMY_H
#define THELIFEENGINECPP_ANATOMY_H

#include <utility>

#ifndef __EMSCRIPTEN_COMPILATION__
#include <boost/unordered_map.hpp>
#else
#include "unordered_map"
#define boost std
#endif

#include <random>
#include <numeric>

#include "Rotation.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "../../Stuff/BlockTypes.hpp"
#include "../../GridStuff/BaseGridBlock.h"
#include "../../PRNGS/lehmer64.h"
#include "../../Stuff/Vector2.h"
//#include "AnatomyContainers.h"
#include "SimpleAnatomyMutationLogic.h"

class Anatomy: public SerializedOrganismStructureContainer {
private:
    SerializedOrganismStructureContainer *
    add_block(BlockTypes type, int block_choice, Rotation rotation, int x_, int y_,
              boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &_organism_blocks,
              std::vector<SerializedAdjacentSpaceContainer> &_single_adjacent_space,
              std::vector<SerializedAdjacentSpaceContainer> &_single_diagonal_adjacent_space,
              boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space);
    SerializedOrganismStructureContainer *change_block(BlockTypes type, int block_choice, lehmer64 *gen);
    SerializedOrganismStructureContainer *remove_block(int block_choice);

    SerializedOrganismStructureContainer *
    make_container(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &_organism_blocks,
                   boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                   ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view *) SW_ORGANISM_BLOCK_NAMES> &_c) const;


    void subtract_difference(int x, int y);
    Vector2<int> recenter_to_imaginary();
    Vector2<int> recenter_to_existing();

public:
    explicit Anatomy(SerializedOrganismStructureContainer *structure);
    Anatomy(const Anatomy& anatomy);
    Anatomy()=default;
    Anatomy & operator=(Anatomy const & other_anatomy);
    Anatomy & operator=(Anatomy && other_anatomy) noexcept ;

    SerializedOrganismStructureContainer * add_random_block(OrganismBlockParameters& bp, lehmer64 &mt);
    SerializedOrganismStructureContainer * change_random_block(OrganismBlockParameters& bp, lehmer64 &gen);
    SerializedOrganismStructureContainer * remove_random_block(lehmer64 &gen);

    void set_block(BlockTypes type, Rotation rotation, int x, int y);
    void set_many_blocks(const std::vector<SerializedOrganismBlockContainer> &blocks);
    Vector2<int> recenter_blocks(bool imaginary_center);
};


#endif //THELIFEENGINECPP_ANATOMY_H
