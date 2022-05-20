//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"

//TODO program will do some redundant work, but i don't know if optimization i can think of will make it much faster
//TODO std::move structure
Anatomy::Anatomy(SerializedOrganismStructureContainer *structure) {
    _organism_blocks = std::move(structure->organism_blocks);

    _producing_space = std::move(structure->producing_space);
    _eating_space    = std::move(structure->eating_space);
    _armor_space     = std::move(structure->armor_space);

    _single_adjacent_space          = std::move(structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::move(structure->single_diagonal_adjacent_space);
    _double_adjacent_space          = std::move(structure->double_adjacent_space);

    _mouth_blocks    = structure->mouth_blocks;
    _producer_blocks = structure->producer_blocks;
    _mover_blocks    = structure->mover_blocks;
    _killer_blocks   = structure->killer_blocks;
    _armor_blocks    = structure->armor_blocks;
    _eye_blocks      = structure->eye_blocks;
    delete structure;
}

Anatomy::Anatomy(const std::shared_ptr<Anatomy>& anatomy) {
    _organism_blocks = std::vector(anatomy->_organism_blocks);

    _producing_space = std::vector(anatomy->_producing_space);
    _eating_space    = std::vector(anatomy->_eating_space);
    _armor_space     = std::vector(anatomy->_armor_space);

    _single_adjacent_space          = std::vector(anatomy->_single_adjacent_space);
    _single_diagonal_adjacent_space = std::vector(anatomy->_single_diagonal_adjacent_space);
    _double_adjacent_space          = std::vector(anatomy->_double_adjacent_space);

    _mouth_blocks    = anatomy->_mouth_blocks;
    _producer_blocks = anatomy->_producer_blocks;
    _mover_blocks    = anatomy->_mover_blocks;
    _killer_blocks   = anatomy->_killer_blocks;
    _armor_blocks    = anatomy->_armor_blocks;
    _eye_blocks      = anatomy->_eye_blocks;
}

Anatomy::~Anatomy()=default;

void Anatomy::set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset))
    {single_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_single_adjacent_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_diagonal_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_adjacent(xmap.first, yxmap.first,  1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first,  0, 1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
        }
    }
}

void Anatomy::create_single_diagonal_adjacent_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
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
                                  boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset)
        ) {double_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_double_adjacent_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
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
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &producing_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
        int32_t producer_blocks) {
    //if (producer_blocks > 0) {
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
    //}
}

void Anatomy::create_eating_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &eating_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                                  int32_t mouth_blocks) {
    //if (mouth_blocks > 0) {
        for (auto &xmap: organism_blocks) {
            for (auto const &yxmap: xmap.second) {
                if (yxmap.second.type == BlockTypes::MouthBlock) {
                    auto x = xmap.first;
                    auto y = yxmap.first;
                    if (single_adjacent_space[x + 1].count(y)) { eating_space[x + 1][y] = true; }
                    if (single_adjacent_space[x - 1].count(y)) { eating_space[x - 1][y] = true; }
                    if (single_adjacent_space[x].count(y + 1)) { eating_space[x][y + 1] = true; }
                    if (single_adjacent_space[x].count(y - 1)) { eating_space[x][y - 1] = true; }
                }
            }
        }
    //}
}

void Anatomy::create_armor_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                 boost::unordered_map<int, boost::unordered_map<int, bool>> &armor_space,
                                 boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                                 int32_t armor_blocks) {
    //if (armor_blocks > 0) {
        for (auto &xmap: organism_blocks) {
            for (auto &yxmap: xmap.second) {
                auto x = xmap.first;
                auto y = yxmap.first;
                if (single_adjacent_space[x + 1].count(y)) {
                    if (yxmap.second.type == BlockTypes::ArmorBlock) {
                        armor_space[x + 1][y] = true;
                    } else {
                        armor_space[x + 1][y] = false;
                    }
                }
                if (single_adjacent_space[x - 1].count(y)) {
                    if (yxmap.second.type == BlockTypes::ArmorBlock) {
                        armor_space[x - 1][y] = true;
                    } else {
                        armor_space[x - 1][y] = false;
                    }
                }
                if (single_adjacent_space[x].count(y + 1)) {
                    if (yxmap.second.type == BlockTypes::ArmorBlock) {
                        armor_space[x][y + 1] = true;
                    } else {
                        armor_space[x][y + 1] = false;
                    }
                }
                if (single_adjacent_space[x].count(y - 1)) {
                    if (yxmap.second.type == BlockTypes::ArmorBlock) {
                        armor_space[x][y - 1] = true;
                    } else {
                        armor_space[x][y - 1] = false;
                    }
                }
            }
        }
    //}
}

SerializedOrganismStructureContainer * Anatomy::serialize(
        const boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& producing_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& armor_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space,
        int32_t mouth_blocks,
        int32_t producer_blocks,
        int32_t mover_blocks,
        int32_t killer_blocks,
        int32_t armor_blocks,
        int32_t eye_blocks) {
    std::vector<SerializedOrganismBlockContainer> _organism_blocks;

    std::vector<SerializedAdjacentSpaceContainer> _producing_space;
    std::vector<SerializedAdjacentSpaceContainer> _eating_space;
    //TODO armor space is redundant. refactor functionality into single_adjacent_space.
    std::vector<SerializedArmorSpaceContainer>    _armor_space;

    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _double_adjacent_space;

    _organism_blocks.reserve(get_map_size(organism_blocks));

    _producing_space.reserve(get_map_size(producing_space));
    _eating_space.reserve(   get_map_size(eating_space));
    _armor_space.reserve(    get_map_size(armor_space));

    _single_adjacent_space.reserve(         get_map_size(single_adjacent_space));
    _single_diagonal_adjacent_space.reserve(get_map_size(single_diagonal_adjacent_space));
    _double_adjacent_space.reserve(         get_map_size(double_adjacent_space));

    //item.first = position in map, item.second = content
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            _organism_blocks.emplace_back(yxmap.second.type, yxmap.second.rotation, xmap.first, yxmap.first,
                                          yxmap.second.neighbors.up, yxmap.second.neighbors.left, yxmap.second.neighbors.down, yxmap.second.neighbors.right);
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

    for (auto const &xmap: armor_space) {
        for (auto const &yxmap: xmap.second) {
            _armor_space.emplace_back(xmap.first, yxmap.first, yxmap.second);
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

    return new SerializedOrganismStructureContainer{_organism_blocks,

                                                    _producing_space,
                                                    _eating_space,
                                                    _armor_space,

                                                    _single_adjacent_space,
                                                    _single_diagonal_adjacent_space,
                                                    _double_adjacent_space,

                                                    mouth_blocks,
                                                    producer_blocks,
                                                    mover_blocks,
                                                    killer_blocks,
                                                    armor_blocks,
                                                    eye_blocks};
}

SerializedOrganismStructureContainer * Anatomy::add_block(BlockTypes type, int block_choice, Rotation rotation, int x_,
                                                          int y_) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> armor_space;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    int x;
    int y;
    if (block_choice > -1) {
        if (block_choice < _single_adjacent_space.size()) {
            x = _single_adjacent_space[block_choice].relative_x;
            y = _single_adjacent_space[block_choice].relative_y;
        } else {
            //TODO VERY IMPORTANT! invalid read here when create_child.
            x = _single_diagonal_adjacent_space[block_choice-_single_adjacent_space.size()].relative_x;
            y = _single_diagonal_adjacent_space[block_choice-_single_adjacent_space.size()].relative_y;
        }
    } else {
        x = x_;
        y = y_;
    }

    auto block = BaseGridBlock();
    block.type     = type;
    block.rotation = rotation;
    organism_blocks[x][y] = block;

    int32_t mouth_blocks    = _mouth_blocks;
    int32_t producer_blocks = _producer_blocks;
    int32_t mover_blocks    = _mover_blocks;
    int32_t killer_blocks   = _killer_blocks;
    int32_t armor_blocks    = _armor_blocks;
    int32_t eye_blocks      = _eye_blocks;

    switch (type) {
        case MouthBlock:    mouth_blocks++    ; break;
        case ProducerBlock: producer_blocks++ ; break;
        case MoverBlock:    mover_blocks++    ; break;
        case KillerBlock:   killer_blocks++   ; break;
        case ArmorBlock:    armor_blocks++    ; break;
        case EyeBlock:      eye_blocks++      ; break;
        case EmptyBlock:
        case FoodBlock:
        case WallBlock:
            break;
    }

    calculate_neighbors(organism_blocks);

    create_single_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_armor_space(    organism_blocks, armor_space,     single_adjacent_space, armor_blocks);

    //TODO VERY IMPORTANT! memory leak here. maybe not
    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     armor_space,
                     single_adjacent_space,
                     single_diagonal_adjacent_space,
                     double_adjacent_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
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

    if (type_choice < block_parameters.MouthBlock.chance_weight)   {return add_block(BlockTypes::MouthBlock,
                                                                                     block_choice, Rotation::UP, 0, 0);}
    type_choice -= block_parameters.MouthBlock.chance_weight;

    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return add_block(BlockTypes::ProducerBlock,
                                                                                      block_choice, Rotation::UP, 0, 0);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;

    if (type_choice < block_parameters.MoverBlock.chance_weight)    {return add_block(BlockTypes::MoverBlock,
                                                                                      block_choice, Rotation::UP, 0, 0);}
    type_choice -= block_parameters.MoverBlock.chance_weight;

    if (type_choice < block_parameters.KillerBlock.chance_weight)   {return add_block(BlockTypes::KillerBlock,
                                                                                      block_choice, Rotation::UP, 0, 0);}
    type_choice -= block_parameters.KillerBlock.chance_weight;

    if (type_choice < block_parameters.ArmorBlock.chance_weight)    {return add_block(BlockTypes::ArmorBlock,
                                                                                      block_choice, Rotation::UP, 0, 0);}
    Rotation rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(mt));
    return add_block(BlockTypes::EyeBlock, block_choice, rotation, 0, 0);
}

SerializedOrganismStructureContainer * Anatomy::change_block(BlockTypes type, int block_choice, std::mt19937 *mt) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> armor_space;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    int32_t mouth_blocks    = _mouth_blocks;
    int32_t producer_blocks = _producer_blocks;
    int32_t mover_blocks    = _mover_blocks;
    int32_t killer_blocks   = _killer_blocks;
    int32_t armor_blocks    = _armor_blocks;
    int32_t eye_blocks      = _eye_blocks;

    switch (_organism_blocks[block_choice].type) {
        case MouthBlock:    mouth_blocks--    ; break;
        case ProducerBlock: producer_blocks-- ; break;
        case MoverBlock:    mover_blocks--    ; break;
        case KillerBlock:   killer_blocks--   ; break;
        case ArmorBlock:    armor_blocks--    ; break;
        case EyeBlock:      eye_blocks--      ; break;
        case EmptyBlock:
        case FoodBlock:
        case WallBlock:
            break;
    }

    switch (type) {
        case MouthBlock:    mouth_blocks++    ; break;
        case ProducerBlock: producer_blocks++ ; break;
        case MoverBlock:    mover_blocks++    ; break;
        case KillerBlock:   killer_blocks++   ; break;
        case ArmorBlock:    armor_blocks++    ; break;
        case EyeBlock:      eye_blocks++      ; break;
        case EmptyBlock:
        case FoodBlock:
        case WallBlock:
            break;
    }

    for (int i = 0; i < _organism_blocks.size(); i ++) {
        organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].type     = _organism_blocks[i].type;
        organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].rotation = _organism_blocks[i].rotation;
        organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].neighbors = _organism_blocks[i].neighbors;
        if (i == block_choice) {
            organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].type = type;
            if (type == BlockTypes::EyeBlock) {
                organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
            }
        }
    }

    create_single_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_armor_space(    organism_blocks, armor_space,     single_adjacent_space, armor_blocks);

    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     armor_space,
                     single_adjacent_space,
                     single_diagonal_adjacent_space,
                     double_adjacent_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
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

    if (type_choice < block_parameters.MouthBlock.chance_weight)    {return change_block(BlockTypes::MouthBlock,
                                                                                         block_choice, &mt);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return change_block(BlockTypes::ProducerBlock,
                                                                                         block_choice, &mt);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight)    {return change_block(BlockTypes::MoverBlock,
                                                                                         block_choice, &mt);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight)   {return change_block(BlockTypes::KillerBlock,
                                                                                         block_choice, &mt);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight)    {return change_block(BlockTypes::ArmorBlock,
                                                                                         block_choice, &mt);}

    return change_block(BlockTypes::EyeBlock, block_choice, &mt);
}

//TODO add constraints
SerializedOrganismStructureContainer * Anatomy::remove_block(int block_choice) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> armor_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;

    int32_t mouth_blocks    = _mouth_blocks;
    int32_t producer_blocks = _producer_blocks;
    int32_t mover_blocks    = _mover_blocks;
    int32_t killer_blocks   = _killer_blocks;
    int32_t armor_blocks    = _armor_blocks;
    int32_t eye_blocks      = _eye_blocks;

    switch (_organism_blocks[block_choice].type) {
        case MouthBlock:    mouth_blocks--    ; break;
        case ProducerBlock: producer_blocks-- ; break;
        case MoverBlock:    mover_blocks--    ; break;
        case KillerBlock:   killer_blocks--   ; break;
        case ArmorBlock:    armor_blocks--    ; break;
        case EyeBlock:      eye_blocks--      ; break;
        case EmptyBlock:
        case FoodBlock:
        case WallBlock:
            break;
    }

    int x = _organism_blocks[block_choice].relative_x;
    int y = _organism_blocks[block_choice].relative_y;

    if (x == 0 && y == 0) {
        //if center block is deleted, we need to recenter the anatomy
        reset_organism_center(_organism_blocks, organism_blocks, x, y);
    } else {
        for (auto &block: _organism_blocks) {
            organism_blocks[block.relative_x][block.relative_y].type     = block.type;
            organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
            organism_blocks[block.relative_x][block.relative_y].neighbors = block.neighbors;
        }
    }

    organism_blocks[x].erase(organism_blocks[x].find(y));

    calculate_neighbors(organism_blocks);

    create_single_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
    create_double_adjacent_space(         organism_blocks, single_adjacent_space, single_diagonal_adjacent_space, double_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_armor_space(    organism_blocks, armor_space,     single_adjacent_space, armor_blocks);

    //Memory leak here
    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     armor_space,
                     single_adjacent_space,
                     single_diagonal_adjacent_space,
                     double_adjacent_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
}

SerializedOrganismStructureContainer * Anatomy::remove_random_block(std::mt19937 &mt) {
    int block_choice = std::uniform_int_distribution<int>{0, int(_organism_blocks.size())-1}(mt);
    return remove_block(block_choice);
}

void Anatomy::reset_organism_center(std::vector<SerializedOrganismBlockContainer> & _organism_blocks,
                                    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                    int & x, int & y) {
    //first we need to find the closest block to the center.
    // just in case...
    int32_t min_x = INT32_MAX; int32_t min_y = INT32_MAX; auto pos = -1;
    for (int i = 0; i < _organism_blocks.size(); i++) {
        //if block is the center, we don't do anything, as it will be deleted.
        if (_organism_blocks[i].relative_x == 0 && _organism_blocks[i].relative_y == 0) {continue;}
        if (std::abs(_organism_blocks[i].relative_x) < min_x &&
            std::abs(_organism_blocks[i].relative_y) < min_y) {
            pos = i;
        }
    }
    // we need to shift coordinates of every block by coordinates of a block chosen as a new center.
    int shift_x = _organism_blocks[pos].relative_x;
    int shift_y = _organism_blocks[pos].relative_y;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x - shift_x][block.relative_y - shift_y].type     = block.type;
        organism_blocks[block.relative_x - shift_x][block.relative_y - shift_y].rotation = block.rotation;
        organism_blocks[block.relative_x - shift_x][block.relative_y - shift_y].neighbors = block.neighbors;
    }
    // new coordinates of a previous center, which will be deleted.
    x -= shift_x;
    y -= shift_y;
}

void Anatomy::set_block(BlockTypes type, Rotation rotation, int x, int y) {
    for (auto & item: _organism_blocks) {
        if (item.relative_x == x && item.relative_y == y) {
            switch (item.type) {
                case MouthBlock:    _mouth_blocks--    ; break;
                case ProducerBlock: _producer_blocks-- ; break;
                case MoverBlock:    _mover_blocks--    ; break;
                case KillerBlock:   _killer_blocks--   ; break;
                case ArmorBlock:    _armor_blocks--    ; break;
                case EyeBlock:      _eye_blocks--      ; break;
                case EmptyBlock:
                case FoodBlock:
                case WallBlock:
                    break;
            }

            switch (type) {
                case MouthBlock:    _mouth_blocks++    ; break;
                case ProducerBlock: _producer_blocks++ ; break;
                case MoverBlock:    _mover_blocks++    ; break;
                case KillerBlock:   _killer_blocks++   ; break;
                case ArmorBlock:    _armor_blocks++    ; break;
                case EyeBlock:      _eye_blocks++      ; break;
                case EmptyBlock:
                case FoodBlock:
                case WallBlock:
                    break;
            }
            item.type = type;
            return;
        }
    }
    auto new_structure = add_block(type, -1, rotation, x, y);

    _organism_blocks = std::move(new_structure->organism_blocks);

    _producing_space = std::move(new_structure->producing_space);
    _eating_space    = std::move(new_structure->eating_space);
    _armor_space     = std::move(new_structure->armor_space);

    _single_adjacent_space          = std::move(new_structure->single_adjacent_space);
    _single_diagonal_adjacent_space = std::move(new_structure->single_diagonal_adjacent_space);
    _double_adjacent_space          = std::move(new_structure->double_adjacent_space);

    _mouth_blocks    = new_structure->mouth_blocks;
    _producer_blocks = new_structure->producer_blocks;
    _mover_blocks    = new_structure->mover_blocks;
    _killer_blocks   = new_structure->killer_blocks;
    _armor_blocks    = new_structure->armor_blocks;
    _eye_blocks      = new_structure->armor_blocks;

    delete new_structure;
}

void Anatomy::calculate_neighbors(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks) {
    for (auto & xmap: organism_blocks) {
        for (auto & xymap: xmap.second) {
            if (organism_blocks[xmap.first+1].count(xymap.first)) {organism_blocks[xmap.first][xymap.first].neighbors.right = true;}
            if (organism_blocks[xmap.first-1].count(xymap.first)) {organism_blocks[xmap.first][xymap.first].neighbors.left  = true;}
            if (organism_blocks[xmap.first].count(xymap.first+1)) {organism_blocks[xmap.first][xymap.first].neighbors.down  = true;}
            if (organism_blocks[xmap.first].count(xymap.first-1)) {organism_blocks[xmap.first][xymap.first].neighbors.up    = true;}
        }
    }
}

template<typename T>
int Anatomy::get_map_size(boost::unordered_map<int, boost::unordered_map<int, T>> map) {
    int total_size = 0;
    for (auto & xmap: map) {
        total_size += xmap.second.size();
    }
    return total_size;
}
