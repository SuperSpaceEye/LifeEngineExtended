//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"
//TODO program will do some redundant work, but i don't know if optimization i can think of will make it much faster

Anatomy::Anatomy(SerializedOrganismStructureContainer *structure) {
    _organism_blocks = std::vector(structure->organism_blocks);
    _single_adjacent_space = std::vector(structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(structure->single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(structure->double_adjacent_space);
    //delete structure;
}

Anatomy::Anatomy(const Anatomy *anatomy) {
    _organism_blocks = std::vector(anatomy->_organism_blocks);
    _single_adjacent_space = std::vector(anatomy->_single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(anatomy->_single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(anatomy->_double_adjacent_space);
}

Anatomy::~Anatomy() {
    delete structure;
}

void Anatomy::set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset))
    {single_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_single_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_adjacent(xmap.first, yxmap.first,  1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first,  0, 1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
        }
    }
}

void Anatomy::create_single_diagonal_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                                    boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                                    boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_adjacent(xmap.first, yxmap.first,  1,  1, organism_blocks, single_diagonal_adjacent_space, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, -1,  1, organism_blocks, single_diagonal_adjacent_space, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, -1, -1, organism_blocks, single_diagonal_adjacent_space, single_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first,  1, -1, organism_blocks, single_diagonal_adjacent_space, single_adjacent_space);
        }
    }
}

void Anatomy::set_double_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset)
        ) {double_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_double_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                           boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    for (auto const &xmap: single_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            set_double_adjacent(xmap.first, yxmap.first,  1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first,  0, 1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
            set_double_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
        }
    }
}

SerializedOrganismStructureContainer * Anatomy::serialize(
        const boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    std::vector<SerializedOrganismBlockContainer> __organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> __single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> __single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> __double_adjacent_space;

    __organism_blocks.reserve(organism_blocks.size());
    __single_adjacent_space.reserve(single_adjacent_space.size());
    __single_diagonal_adjacent_space.reserve(single_diagonal_adjacent_space.size());
    __double_adjacent_space.reserve(double_adjacent_space.size());

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

    for (auto const &xmap: single_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            __single_diagonal_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: double_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            __double_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    return new SerializedOrganismStructureContainer{__organism_blocks, __single_adjacent_space, __single_diagonal_adjacent_space, __double_adjacent_space};
}

SerializedOrganismStructureContainer * Anatomy::add_block(BlockTypes type, int block_choice, int x_, int y_) {
    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
    }
    //TODO possible optimizations here
//    for (auto& block: _single_adjacent_space) {
//        single_adjacent_space[block.relative_x][block.relative_y] = true;
//    }
//    for (auto& block: _single_diagonal_adjacent_space) {
//        single_diagonal_adjacent_space[block.relative_x][block.relative_y] = true;
//    }
//    for (auto& block: _double_adjacent_space) {
//        double_adjacent_space[block.relative_x][block.relative_y] = true;
//    }

    int x;
    int y;
    if (block_choice > -1) {
        if (block_choice < _single_adjacent_space.size()) {
            x = _single_adjacent_space[block_choice].relative_x;
            y = _single_adjacent_space[block_choice].relative_y;
        } else {
            x = _single_diagonal_adjacent_space[block_choice].relative_x;
            y = _single_diagonal_adjacent_space[block_choice].relative_y;
        }
    } else {
        x = x_;
        y = y_;
    }

    auto block = OrganismBlock();
    block.type = type;

    organism_blocks[x][y] = block;
    create_single_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
    return serialize(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
}

SerializedOrganismStructureContainer * Anatomy::add_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt) {
    float total_chance = 0;
    total_chance += block_parameters.MouthBlock   .chance_weight;
    total_chance += block_parameters.ProducerBlock.chance_weight;
    total_chance += block_parameters.MoverBlock   .chance_weight;
    total_chance += block_parameters.KillerBlock  .chance_weight;
    total_chance += block_parameters.ArmorBlock   .chance_weight;
    total_chance += block_parameters.EyeBlock     .chance_weight;

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(mt);
    int   block_choice = std::uniform_int_distribution<int>{0, int(_single_adjacent_space.size()+_single_diagonal_adjacent_space.size())-1}(mt);

    if (type_choice < block_parameters.MouthBlock.chance_weight) {return add_block(BlockTypes::MouthBlock, block_choice);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return add_block(BlockTypes::ProducerBlock, block_choice);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight) {return add_block(BlockTypes::MoverBlock, block_choice);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight) {return add_block(BlockTypes::KillerBlock, block_choice);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight) {return add_block(BlockTypes::ArmorBlock, block_choice);}

    return add_block(BlockTypes::EyeBlock, block_choice);
}

SerializedOrganismStructureContainer * Anatomy::change_block(BlockTypes type, int block_choice) {
    std::vector<SerializedOrganismBlockContainer> __organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> __single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> __single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> __double_adjacent_space;

    __organism_blocks.reserve(_organism_blocks.size());
    __single_adjacent_space.reserve(_single_adjacent_space.size());
    __single_diagonal_adjacent_space.reserve(_single_diagonal_adjacent_space.size());
    __double_adjacent_space.reserve(_double_adjacent_space.size());

    //is this okay to do? idk, but i'll see if this works.
    for (auto const & item: _organism_blocks) {
        __organism_blocks.emplace_back(item);
    }
    for (auto const & item: _single_adjacent_space) {
        __single_adjacent_space.emplace_back(item);
    }
    for (auto const & item: _single_diagonal_adjacent_space) {
        __single_diagonal_adjacent_space.emplace_back(item);
    }
    for (auto const & item: _double_adjacent_space) {
        __double_adjacent_space.emplace_back(item);
    }

    __organism_blocks[block_choice].organism_block.type = type;
    return new SerializedOrganismStructureContainer{__organism_blocks, __single_adjacent_space, __single_diagonal_adjacent_space, __double_adjacent_space};
}

SerializedOrganismStructureContainer * Anatomy::change_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt) {
    float total_chance = 0;
    total_chance += block_parameters.MouthBlock   .chance_weight;
    total_chance += block_parameters.ProducerBlock.chance_weight;
    total_chance += block_parameters.MoverBlock   .chance_weight;
    total_chance += block_parameters.KillerBlock  .chance_weight;
    total_chance += block_parameters.ArmorBlock   .chance_weight;
    total_chance += block_parameters.EyeBlock     .chance_weight;

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(mt);
    int   block_choice = std::uniform_int_distribution<int>{0, int(_organism_blocks.size())-1}(mt);

    if (type_choice < block_parameters.MouthBlock.chance_weight)    {return change_block(BlockTypes::MouthBlock, block_choice);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return change_block(BlockTypes::ProducerBlock, block_choice);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight)    {return change_block(BlockTypes::MoverBlock, block_choice);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight)   {return change_block(BlockTypes::KillerBlock, block_choice);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight)    {return change_block(BlockTypes::ArmorBlock, block_choice);}

    return change_block(BlockTypes::EyeBlock, block_choice);
}

//TODO add constraints
SerializedOrganismStructureContainer * Anatomy::remove_block(int block_choice) {
    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
    }
    //TODO and here
//    for (auto& block: _single_adjacent_space) {
//        single_adjacent_space[block.relative_x][block.relative_y] = true;
//    }
//    for (auto& block: _single_diagonal_adjacent_space) {
//        single_diagonal_adjacent_space[block.relative_x][block.relative_y] = true;
//    }
//    for (auto& block: _double_adjacent_space) {
//        double_adjacent_space[block.relative_x][block.relative_y] = true;
//    }

    int x = _organism_blocks[block_choice].relative_x;
    int y = _organism_blocks[block_choice].relative_y;

    organism_blocks[x].erase(organism_blocks[x].find(y));

    create_single_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
    return serialize(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
}

SerializedOrganismStructureContainer * Anatomy::remove_random_block(std::mt19937 &mt) {
    int block_choice = std::uniform_int_distribution<int>{0, int(_organism_blocks.size())-1}(mt);
    return remove_block(block_choice);
}

void Anatomy::set_block(BlockTypes type, int x, int y) {
    for (auto & item: _organism_blocks) {
        if (item.relative_x == x && item.relative_y == y) {
            item.organism_block.type = type;
            return;
        }
    }
    auto new_structure = add_block(type, -1, x, y);
    delete structure;
    structure = new_structure;

    _organism_blocks = std::vector(structure->organism_blocks);
    _single_adjacent_space = std::vector(structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(structure->single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(structure->double_adjacent_space);
}