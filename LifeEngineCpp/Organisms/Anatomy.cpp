//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"
//TODO program will do some redundant work, but i don't know if optimization i can think of will make it much faster

Anatomy::Anatomy(const OrganismStructureContainer structure) {
    _organism_blocks       = structure.organism_blocks;
    _single_adjacent_space = structure.single_adjacent_space;
    _double_adjacent_space = structure.double_adjacent_space;
}

void Anatomy::set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)) {single_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_single_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_adjacent(xmap.first, yxmap.first, 1, 0, organism_blocks, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, 0, 1, organism_blocks, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space);
        }
    }
}

void Anatomy::set_double_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset) && !single_adjacent_space[x+x_offset].count(y+y_offset)) {double_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_double_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    for (auto const &xmap: single_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            set_double_adjacent(xmap.first, yxmap.first, 1, 0, organism_blocks, single_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first, 0, 1, organism_blocks, single_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space, double_adjacent_space);
        }
    }
}

OrganismStructureContainer Anatomy::add_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt) {
    float total_chance = 0;
    total_chance += block_parameters.MouthBlock   .chance_weight;
    total_chance += block_parameters.ProducerBlock.chance_weight;
    total_chance += block_parameters.MoverBlock   .chance_weight;
    total_chance += block_parameters.KillerBlock  .chance_weight;
    total_chance += block_parameters.ArmorBlock   .chance_weight;
    total_chance += block_parameters.EyeBlock     .chance_weight;

//    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(mt);
    int   block_choice = std::uniform_int_distribution<int>{0, int(_single_adjacent_space.size())-1}(mt);

//    for (auto& block: _organism_blocks) {
//        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
//    }
//    for (auto& block: _single_adjacent_space) {
//        single_adjacent_space[block.relative_x][block.relative_y] = true;
//    }
//    for (auto& block: _double_adjacent_space) {
//        double_adjacent_space[block.relative_x][block.relative_y] = true;
//    }

    if (type_choice < block_parameters.MouthBlock.chance_weight) {return add_block(BlockTypes::MouthBlock, block_choice);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return add_block(BlockTypes::ProducerBlock, block_choice);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight) {return add_block(BlockTypes::MoverBlock, block_choice);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight) {return add_block(BlockTypes::KillerBlock, block_choice);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight) {return add_block(BlockTypes::ArmorBlock, block_choice);}

    {return add_block(BlockTypes::EyeBlock, block_choice);}
}

OrganismStructureContainer Anatomy::serialize(
        boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space) {
    std::vector<OrganismBlockContainer> __organism_blocks;
    std::vector<AdjacentSpaceContainer> __single_adjacent_space;
    std::vector<AdjacentSpaceContainer> __double_adjacent_space;

    __organism_blocks.resize(organism_blocks.size());
    __single_adjacent_space.resize(single_adjacent_space.size());
    __double_adjacent_space.resize(double_adjacent_space.size());

    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            __organism_blocks.emplace_back(yxmap.second, xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: single_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            __single_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: double_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            __double_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    return OrganismStructureContainer{__organism_blocks, __single_adjacent_space, __double_adjacent_space};
}

OrganismStructureContainer Anatomy::add_block(BlockTypes type, int block_choice) {
    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
    }
    for (auto& block: _single_adjacent_space) {
        single_adjacent_space[block.relative_x][block.relative_y] = true;
    }
    for (auto& block: _double_adjacent_space) {
        double_adjacent_space[block.relative_x][block.relative_y] = true;
    }

    int x = _single_adjacent_space[block_choice].relative_x;
    int y = _single_adjacent_space[block_choice].relative_y;

    auto block = OrganismBlock();
    block.type = type;

    organism_blocks[x][y] = block;
    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_double_adjacent_space(organism_blocks, single_adjacent_space, double_adjacent_space);
    return serialize(organism_blocks, single_adjacent_space, double_adjacent_space);
}