//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"
//TODO program will do some redundant work, but i don't know if optimization i can think of will make it much faster

Anatomy::Anatomy(SerializedOrganismStructureContainer *structure) {
    _organism_blocks = std::vector(structure->organism_blocks);

    _producing_space = std::vector(structure->producing_space);
    _eating_space = std::vector(structure->eating_space);

    _single_adjacent_space = std::vector(structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(structure->single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(structure->double_adjacent_space);
    _producers = structure->producers;
    _movers = structure->movers;
    //delete structure;
}

Anatomy::Anatomy(const Anatomy *anatomy) {
    _organism_blocks = std::vector(anatomy->_organism_blocks);

    _producing_space = std::vector(anatomy->_producing_space);
    _eating_space = std::vector(anatomy->_eating_space);

    _single_adjacent_space = std::vector(anatomy->_single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(anatomy->_single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(anatomy->_double_adjacent_space);
    _producers = anatomy->_producers;
    _movers = anatomy->_movers;
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

void Anatomy::create_producing_space(
        boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &producing_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
        const int32_t producers) {
    if (producers > 0) {
        for (auto &xmap: organism_blocks) {
            for (auto const &yxmap: xmap.second) {
                if (yxmap.second.type == BlockTypes::ProducerBlock) {
                    auto x = xmap.first;
                    auto y = yxmap.first;
                    if (single_adjacent_space[x + 1].count(y)) { producing_space[x + 1][y] = true; }
                    if (single_adjacent_space[x - 1].count(y)) { producing_space[x - 1][y] = true; }
                    if (single_adjacent_space[x].count(y + 1)) { producing_space[x][y + 1] = true; }
                    if (single_adjacent_space[x].count(y - 1)) { producing_space[x][y - 1] = true; }
                }
            }
        }
    }
}

void Anatomy::create_eating_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &eating_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space) {
    for (auto &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            if (yxmap.second.type == BlockTypes::MouthBlock) {
                auto x = xmap.first;
                auto y = yxmap.first;
                if (single_adjacent_space[x + 1].count(y)) {eating_space[x + 1][y] = true;}
                if (single_adjacent_space[x - 1].count(y)) {eating_space[x - 1][y] = true;}
                if (single_adjacent_space[x].count(y + 1)) {eating_space[x][y + 1] = true;}
                if (single_adjacent_space[x].count(y - 1)) {eating_space[x][y - 1] = true;}
            }
        }
    }
}

SerializedOrganismStructureContainer * Anatomy::serialize(
        const boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& producing_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space,
        const int32_t producers, const int32_t movers) {
    std::vector<SerializedOrganismBlockContainer> _organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> _producing_space;
    std::vector<SerializedAdjacentSpaceContainer> _eating_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _double_adjacent_space;

    _organism_blocks.reserve(organism_blocks.size());
    _producing_space.reserve(producing_space.size());
    _eating_space.reserve(eating_space.size());
    _single_adjacent_space.reserve(single_adjacent_space.size());
    _single_diagonal_adjacent_space.reserve(single_diagonal_adjacent_space.size());
    _double_adjacent_space.reserve(double_adjacent_space.size());

    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            _organism_blocks.emplace_back(yxmap.second, xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: producing_space) {
        for (auto const &yxmap: xmap.second) {
            _producing_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: eating_space) {
        for (auto const &yxmap: xmap.second) {
            _eating_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: single_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            _single_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: single_diagonal_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            _single_diagonal_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: double_adjacent_space) {
        for (auto const &yxmap: xmap.second) {
            _double_adjacent_space.emplace_back(xmap.first, yxmap.first);
        }
    }

//    std::cout << producers << " " << _producing_space.size() << "\n";
    return new SerializedOrganismStructureContainer{_organism_blocks,
                                                    _producing_space,
                                                    _eating_space,
                                                    _single_adjacent_space,
                                                    _single_diagonal_adjacent_space,
                                                    _double_adjacent_space,
                                                    producers, movers};
}

SerializedOrganismStructureContainer * Anatomy::add_block(BlockTypes type, int block_choice, int x_, int y_) {
    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
    }

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

    if (type == BlockTypes::ProducerBlock) {_producers++;}
    if (type == BlockTypes::MoverBlock) {_movers++;}

    organism_blocks[x][y] = block;
    create_single_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
    create_producing_space(organism_blocks, producing_space, single_adjacent_space, _producers);
    create_eating_space(organism_blocks, eating_space, single_adjacent_space);
    return serialize(organism_blocks, producing_space, eating_space, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space, _producers, _movers);
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
    std::vector<SerializedOrganismBlockContainer> organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> producing_space;
    std::vector<SerializedAdjacentSpaceContainer> eating_space;
    std::vector<SerializedAdjacentSpaceContainer> single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> double_adjacent_space;

    //TODO refactor
    if (_organism_blocks[block_choice].organism_block.type != BlockTypes::ProducerBlock && type == BlockTypes::ProducerBlock) {
        _producers--;
        for (int i = 0; i < _producing_space.size(); i++) {
            if((_organism_blocks[block_choice].relative_x-1 == _producing_space[i].relative_x && _organism_blocks[block_choice].relative_y == _producing_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x+1 == _producing_space[i].relative_x && _organism_blocks[block_choice].relative_y == _producing_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x == _producing_space[i].relative_x-1 && _organism_blocks[block_choice].relative_y == _producing_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x == _producing_space[i].relative_x+1 && _organism_blocks[block_choice].relative_y == _producing_space[i].relative_y)) {
                _producing_space.erase(_producing_space.begin() + i);
                break;
            }
        }
    }

    if (_organism_blocks[block_choice].organism_block.type != BlockTypes::MouthBlock && type == BlockTypes::MouthBlock) {
        for (int i = 0; i < _eating_space.size(); i++) {
            if((_organism_blocks[block_choice].relative_x-1 == _eating_space[i].relative_x && _organism_blocks[block_choice].relative_y == _eating_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x+1 == _eating_space[i].relative_x && _organism_blocks[block_choice].relative_y == _eating_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x == _eating_space[i].relative_x-1 && _organism_blocks[block_choice].relative_y == _eating_space[i].relative_y) ||
               (_organism_blocks[block_choice].relative_x == _eating_space[i].relative_x+1 && _organism_blocks[block_choice].relative_y == _eating_space[i].relative_y)) {
                _producing_space.erase(_eating_space.begin() + i);
                break;
            }
        }
    }

    organism_blocks.reserve(_organism_blocks.size());
    producing_space.reserve(_producing_space.size());
    eating_space.reserve(_eating_space.size());
    single_adjacent_space.reserve(_single_adjacent_space.size());
    single_diagonal_adjacent_space.reserve(_single_diagonal_adjacent_space.size());
    double_adjacent_space.reserve(_double_adjacent_space.size());

    //is this okay to do? idk, but i'll see if this works.
    for (auto const & item: _organism_blocks) {
        organism_blocks.emplace_back(item);
    }
    for (auto const & item: _producing_space) {
        producing_space.emplace_back(item);
    }
    for (auto const & item: _eating_space) {
        eating_space.emplace_back(item);
    }
    for (auto const & item: _single_adjacent_space) {
        single_adjacent_space.emplace_back(item);
    }
    for (auto const & item: _single_diagonal_adjacent_space) {
        single_diagonal_adjacent_space.emplace_back(item);
    }
    for (auto const & item: _double_adjacent_space) {
        double_adjacent_space.emplace_back(item);
    }

    organism_blocks[block_choice].organism_block.type = type;
    return new SerializedOrganismStructureContainer{organism_blocks,
                                                    producing_space,
                                                    eating_space,
                                                    single_adjacent_space,
                                                    single_diagonal_adjacent_space,
                                                    double_adjacent_space,
                                                    _producers, _movers};
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
    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y] = block.organism_block;
    }

    int x = _organism_blocks[block_choice].relative_x;
    int y = _organism_blocks[block_choice].relative_y;

    organism_blocks[x].erase(organism_blocks[x].find(y));

    create_single_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);
    create_producing_space(organism_blocks, producing_space, single_adjacent_space, _producers);
    create_eating_space(organism_blocks, eating_space, single_adjacent_space);
    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     single_adjacent_space,
                     single_diagonal_adjacent_space,
                     double_adjacent_space,
                     _producers, _movers);
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
    _producing_space = std::vector(structure->producing_space);
    _eating_space = std::vector(structure->eating_space);
    _single_adjacent_space = std::vector(structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(structure->single_diagonal_adjacent_space);
    _double_adjacent_space = std::vector(structure->double_adjacent_space);
    _producers = structure->producers;
    _movers = structure->movers;
}