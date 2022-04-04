//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ANATOMY_H
#define THELIFEENGINECPP_ANATOMY_H

#include <utility>

#include "boost/unordered_map.hpp"
#include "Organism_parts/OrganismBlock.h"
#include "random"
#include "../OrganismBlockParameters.h"
#include "../BlockTypes.h"

struct OrganismBlockContainer {
    OrganismBlock organism_block;
    int relative_x;
    int relative_y;

    OrganismBlockContainer(OrganismBlock organism_block,int relative_x,int relative_y):
    organism_block(organism_block), relative_x(relative_x), relative_y(relative_y) {}
};

struct AdjacentSpaceContainer {
    int relative_x;
    int relative_y;

    AdjacentSpaceContainer(int relative_x, int relative_y):
    relative_x(relative_x), relative_y(relative_y) {}
};

struct OrganismStructureContainer {
    std::vector<OrganismBlockContainer> organism_blocks;
    std::vector<AdjacentSpaceContainer> single_adjacent_space;
    std::vector<AdjacentSpaceContainer> double_adjacent_space;

    OrganismStructureContainer(
            std::vector<OrganismBlockContainer> organism_blocks,
            std::vector<AdjacentSpaceContainer> single_adjacent_space,
            std::vector<AdjacentSpaceContainer> double_adjacent_space):
    organism_blocks(      std::move(organism_blocks)),
    single_adjacent_space(std::move(single_adjacent_space)),
    double_adjacent_space(std::move(double_adjacent_space)) {}
};

class Anatomy {
private:
    void set_block(int x, int y, OrganismBlock block, OrganismBlockContainer& container);

    void set_single_adjacent(int x, int y, int x_offset, int y_offset,
                             boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space);
    void set_double_adjacent(int x, int y, int x_offset, int y_offset,
                             boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space);

    OrganismStructureContainer serialize(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks,
                                         boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space,
                                         boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space);

    void create_single_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space);
    void create_double_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space);

    OrganismStructureContainer add_block(BlockTypes type, int block_choice);

public:
    std::vector<OrganismBlockContainer> _organism_blocks;
    std::vector<AdjacentSpaceContainer> _single_adjacent_space;
    std::vector<AdjacentSpaceContainer> _double_adjacent_space;

    Anatomy(OrganismStructureContainer structure);

    OrganismStructureContainer add_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt);
    OrganismStructureContainer change_random_block();
    OrganismStructureContainer delete_random_block();
};


#endif //THELIFEENGINECPP_ANATOMY_H
