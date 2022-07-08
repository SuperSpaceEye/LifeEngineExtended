//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"

Anatomy::Anatomy(SerializedOrganismStructureContainer *structure) {
    _organism_blocks = std::move(structure->organism_blocks);

    _producing_space = std::move(structure->producing_space);
    _eating_space    = std::move(structure->eating_space);
    _killing_space   = std::move(structure->killing_space);

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
    _killing_space   = std::vector(anatomy->_killing_space);

    _mouth_blocks    = anatomy->_mouth_blocks;
    _producer_blocks = anatomy->_producer_blocks;
    _mover_blocks    = anatomy->_mover_blocks;
    _killer_blocks   = anatomy->_killer_blocks;
    _armor_blocks    = anatomy->_armor_blocks;
    _eye_blocks      = anatomy->_eye_blocks;
}

void Anatomy::set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                                  const BaseGridBlock &block) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset)) {
        single_adjacent_space[x + x_offset][y + y_offset] = true;
    }
}

void Anatomy::create_single_adjacent_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_adjacent(xmap.first, yxmap.first,  1, 0, organism_blocks, single_adjacent_space, yxmap.second);
            set_single_adjacent(xmap.first, yxmap.first, -1, 0, organism_blocks, single_adjacent_space, yxmap.second);
            set_single_adjacent(xmap.first, yxmap.first, 0,  1, organism_blocks, single_adjacent_space, yxmap.second);
            set_single_adjacent(xmap.first, yxmap.first, 0, -1, organism_blocks, single_adjacent_space, yxmap.second);
        }
    }
}

void Anatomy::set_single_diagonal_adjacent(int x, int y, int x_offset, int y_offset,
                                  boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset))
    {single_diagonal_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void Anatomy::create_single_diagonal_adjacent_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            set_single_diagonal_adjacent(xmap.first, yxmap.first, 1,   1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_diagonal_adjacent(xmap.first, yxmap.first, -1,  1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_diagonal_adjacent(xmap.first, yxmap.first, -1, -1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
            set_single_diagonal_adjacent(xmap.first, yxmap.first, 1,  -1, organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);
        }
    }
}

void Anatomy::create_producing_space(
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> &producing_space,
        boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
        std::vector<int> & num_producing_space,
        int32_t producer_blocks) {
    if (producer_blocks > 0) {
        num_producing_space.resize(producer_blocks, 0);
        int i = -1;
        for (auto &xmap: organism_blocks) {
            int ii = 0;
            for (auto const &yxmap: xmap.second) {
                if (yxmap.second.type == BlockTypes::ProducerBlock) {
                    i++;
                    auto x = xmap.first;
                    auto y = yxmap.first;
                    if (single_adjacent_space[x + 1].count(y)) { producing_space[x + 1][y] = ProducerAdjacent{i}; ii++;}
                    if (single_adjacent_space[x - 1].count(y)) { producing_space[x - 1][y] = ProducerAdjacent{i}; ii++;}
                    if (single_adjacent_space[x].count(y + 1)) { producing_space[x][y + 1] = ProducerAdjacent{i}; ii++;}
                    if (single_adjacent_space[x].count(y - 1)) { producing_space[x][y - 1] = ProducerAdjacent{i}; ii++;}
                    num_producing_space[i] = ii;
                }
            }
        }
    }
}

void Anatomy::create_eating_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>> &eating_space,
                                  boost::unordered_map<int, boost::unordered_map<int, bool>>&single_adjacent_space,
                                  int32_t mouth_blocks) {
    if (mouth_blocks > 0) {
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
    }
}

void Anatomy::create_killing_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                 boost::unordered_map<int, boost::unordered_map<int, bool>>& killing_space,
                                 boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                 int32_t killer_blocks) {
    if (killer_blocks > 0) {
        for (auto &xmap: organism_blocks) {
            for (auto const &yxmap: xmap.second) {
                if (yxmap.second.type == BlockTypes::KillerBlock) {
                    auto x = xmap.first;
                    auto y = yxmap.first;
                    if (single_adjacent_space[x + 1].count(y)) { killing_space[x + 1][y] = true; }
                    if (single_adjacent_space[x - 1].count(y)) { killing_space[x - 1][y] = true; }
                    if (single_adjacent_space[x].count(y + 1)) { killing_space[x][y + 1] = true; }
                    if (single_adjacent_space[x].count(y - 1)) { killing_space[x][y - 1] = true; }
                }
            }
        }
    }
}

SerializedOrganismStructureContainer * Anatomy::serialize(
        const boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        const boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>>& producing_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
        const boost::unordered_map<int, boost::unordered_map<int, bool>>& killing_space,

        const std::vector<int> & num_producing_space,
        int32_t mouth_blocks,
        int32_t producer_blocks,
        int32_t mover_blocks,
        int32_t killer_blocks,
        int32_t armor_blocks,
        int32_t eye_blocks) {
    std::vector<SerializedOrganismBlockContainer> _organism_blocks;

    std::vector<std::vector<SerializedAdjacentSpaceContainer>> _producing_space;
    std::vector<SerializedAdjacentSpaceContainer> _eating_space;
    std::vector<SerializedAdjacentSpaceContainer> _killing_space;

    _organism_blocks.reserve(get_map_size(organism_blocks));

    _producing_space.reserve(producer_blocks);
    _eating_space.reserve(   get_map_size(eating_space));
    _killing_space.reserve(get_map_size(killing_space));

    //item.first = position in map, item.second = content
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            _organism_blocks.emplace_back(yxmap.second.type, yxmap.second.rotation, xmap.first, yxmap.first);
        }
    }

    for (int i = 0; i < num_producing_space.size(); i++) {
        _producing_space.emplace_back(std::vector<SerializedAdjacentSpaceContainer>());
        _producing_space[i].reserve(num_producing_space[i]);
    }

    for (auto const &xmap: producing_space) {
        for (auto const &yxmap: xmap.second) {
            _producing_space[yxmap.second.producer].emplace_back(SerializedAdjacentSpaceContainer(xmap.first, yxmap.first));
        }
    }

    for (auto const &xmap: eating_space) {
        for (auto const &yxmap: xmap.second) {
            _eating_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    for (auto const &xmap: killing_space) {
        for (auto const &yxmap: xmap.second) {
            _killing_space.emplace_back(xmap.first, yxmap.first);
        }
    }

    return new SerializedOrganismStructureContainer{_organism_blocks,

                                                    _producing_space,
                                                    _eating_space,
                                                    _killing_space,

                                                    mouth_blocks,
                                                    producer_blocks,
                                                    mover_blocks,
                                                    killer_blocks,
                                                    armor_blocks,
                                                    eye_blocks};
}

SerializedOrganismStructureContainer * Anatomy::add_block(BlockTypes type, int block_choice, Rotation rotation, int x_,
                                                          int y_,
                                                          boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                                          std::vector<SerializedAdjacentSpaceContainer> &_single_adjacent_space,
                                                          std::vector<SerializedAdjacentSpaceContainer> &_single_diagonal_adjacent_space,
                                                          boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space) {
    boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> killing_space;

    std::vector<int> num_producing_space;

    int x;
    int y;
    if (block_choice > -1) {
        if (block_choice < _single_adjacent_space.size()) {
            x = _single_adjacent_space[block_choice].relative_x;
            y = _single_adjacent_space[block_choice].relative_y;
        } else {
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

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_killing_space(  organism_blocks, killing_space,   single_adjacent_space, killer_blocks);

    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     killing_space,
                     num_producing_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
}

SerializedOrganismStructureContainer * Anatomy::add_random_block(OrganismBlockParameters& block_parameters, lehmer64 &mt) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;

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
                                                                                     block_choice, Rotation::UP, 0, 0,
                                                                                     organism_blocks, _single_adjacent_space,
                                                                                     _single_diagonal_adjacent_space, single_adjacent_space);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return add_block(BlockTypes::ProducerBlock,
                                                                                      block_choice, Rotation::UP, 0, 0,
                                                                                      organism_blocks, _single_adjacent_space,
                                                                                      _single_diagonal_adjacent_space, single_adjacent_space);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight)    {return add_block(BlockTypes::MoverBlock,
                                                                                      block_choice, Rotation::UP, 0, 0,
                                                                                      organism_blocks, _single_adjacent_space,
                                                                                      _single_diagonal_adjacent_space, single_adjacent_space);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight)   {return add_block(BlockTypes::KillerBlock,
                                                                                      block_choice, Rotation::UP, 0, 0,
                                                                                      organism_blocks, _single_adjacent_space,
                                                                                      _single_diagonal_adjacent_space, single_adjacent_space);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight)    {return add_block(BlockTypes::ArmorBlock,
                                                                                      block_choice, Rotation::UP, 0, 0,
                                                                                      organism_blocks, _single_adjacent_space,
                                                                                      _single_diagonal_adjacent_space, single_adjacent_space);}
    Rotation rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(mt));
    return add_block(BlockTypes::EyeBlock, block_choice, rotation, 0, 0,
                     organism_blocks, _single_adjacent_space,
                     _single_diagonal_adjacent_space, single_adjacent_space);
}

SerializedOrganismStructureContainer * Anatomy::change_block(BlockTypes type, int block_choice, lehmer64 *mt) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> killing_space;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    std::vector<int> num_producing_space;

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
        organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].type      = _organism_blocks[i].type;
        organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].rotation  = _organism_blocks[i].rotation;
        if (i == block_choice) {
            organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].type = type;
            if (type == BlockTypes::EyeBlock) {
                organism_blocks[_organism_blocks[i].relative_x][_organism_blocks[i].relative_y].rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
            }
        }
    }

    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_killing_space(  organism_blocks, killing_space,   single_adjacent_space, killer_blocks);

    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     killing_space,
                     num_producing_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
}

SerializedOrganismStructureContainer * Anatomy::change_random_block(OrganismBlockParameters& block_parameters, lehmer64 &gen) {
    float total_chance = 0;
    total_chance += block_parameters.MouthBlock   .chance_weight;
    total_chance += block_parameters.ProducerBlock.chance_weight;
    total_chance += block_parameters.MoverBlock   .chance_weight;
    total_chance += block_parameters.KillerBlock  .chance_weight;
    total_chance += block_parameters.ArmorBlock   .chance_weight;
    total_chance += block_parameters.EyeBlock     .chance_weight;

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(gen);
    int   block_choice = std::uniform_int_distribution<int>{0, int(_organism_blocks.size())-1}(gen);

    if (type_choice < block_parameters.MouthBlock.chance_weight)    {return change_block(BlockTypes::MouthBlock,
                                                                                         block_choice, &gen);}
    type_choice -= block_parameters.MouthBlock.chance_weight;
    if (type_choice < block_parameters.ProducerBlock.chance_weight) {return change_block(BlockTypes::ProducerBlock,
                                                                                         block_choice, &gen);}
    type_choice -= block_parameters.ProducerBlock.chance_weight;
    if (type_choice < block_parameters.MoverBlock.chance_weight)    {return change_block(BlockTypes::MoverBlock,
                                                                                         block_choice, &gen);}
    type_choice -= block_parameters.MoverBlock.chance_weight;
    if (type_choice < block_parameters.KillerBlock.chance_weight)   {return change_block(BlockTypes::KillerBlock,
                                                                                         block_choice, &gen);}
    type_choice -= block_parameters.KillerBlock.chance_weight;
    if (type_choice < block_parameters.ArmorBlock.chance_weight)    {return change_block(BlockTypes::ArmorBlock,
                                                                                         block_choice, &gen);}

    return change_block(BlockTypes::EyeBlock, block_choice, &gen);
}

SerializedOrganismStructureContainer * Anatomy::remove_block(int block_choice) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> killing_space;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    std::vector<int> num_producing_space;

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
        }
    }

    organism_blocks[x].erase(organism_blocks[x].find(y));

    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, mouth_blocks);
    create_killing_space(  organism_blocks, killing_space,   single_adjacent_space, killer_blocks);

    return serialize(organism_blocks,
                     producing_space,
                     eating_space,
                     killing_space,
                     num_producing_space,

                     mouth_blocks,
                     producer_blocks,
                     mover_blocks,
                     killer_blocks,
                     armor_blocks,
                     eye_blocks);
}

SerializedOrganismStructureContainer * Anatomy::remove_random_block(lehmer64 &gen) {
    int block_choice = std::uniform_int_distribution<int>{0, int(_organism_blocks.size())-1}(gen);
    return remove_block(block_choice);
}

void Anatomy::reset_organism_center(std::vector<SerializedOrganismBlockContainer> & _organism_blocks,
                                    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                    int & x, int & y) {
    //first we need to find the closest block to the center.
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

    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;

    for (auto& block: _organism_blocks) {
        organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;

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

    auto *new_structure = add_block(type, -1, rotation, x, y,
                                   organism_blocks, _single_adjacent_space,
                                   _single_diagonal_adjacent_space, single_adjacent_space);

    _organism_blocks = std::move(new_structure->organism_blocks);

    _producing_space = std::move(new_structure->producing_space);
    _eating_space    = std::move(new_structure->eating_space);

    _mouth_blocks    = new_structure->mouth_blocks;
    _producer_blocks = new_structure->producer_blocks;
    _mover_blocks    = new_structure->mover_blocks;
    _killer_blocks   = new_structure->killer_blocks;
    _armor_blocks    = new_structure->armor_blocks;
    _eye_blocks      = new_structure->eye_blocks;

    delete new_structure;
}

void Anatomy::set_many_blocks(std::vector<SerializedOrganismBlockContainer> &blocks) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;
    boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> producing_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> eating_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> killing_space;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    std::vector<int> num_producing_space;

    for (auto & block: blocks) {
        switch (block.type) {
            case MouthBlock:    _mouth_blocks++    ; break;
            case ProducerBlock: _producer_blocks++ ; break;
            case MoverBlock:    _mover_blocks++    ; break;
            case KillerBlock:   _killer_blocks++   ; break;
            case ArmorBlock:    _armor_blocks++    ; break;
            case EyeBlock:      _eye_blocks++      ; break;
            default: break;
        }

        organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    create_single_adjacent_space(organism_blocks, single_adjacent_space);
    create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, _producer_blocks);
    create_eating_space(   organism_blocks, eating_space,    single_adjacent_space, _mouth_blocks);
    create_killing_space(  organism_blocks, killing_space,   single_adjacent_space, _killer_blocks);

    auto *new_structure = serialize(organism_blocks,
                               producing_space,
                               eating_space,
                               killing_space,
                               num_producing_space,

                               _mouth_blocks,
                               _producer_blocks,
                               _mover_blocks,
                               _killer_blocks,
                               _armor_blocks,
                               _eye_blocks);

    _organism_blocks = std::move(new_structure->organism_blocks);

    _producing_space = std::move(new_structure->producing_space);
    _eating_space    = std::move(new_structure->eating_space);

    _mouth_blocks    = new_structure->mouth_blocks;
    _producer_blocks = new_structure->producer_blocks;
    _mover_blocks    = new_structure->mover_blocks;
    _killer_blocks   = new_structure->killer_blocks;
    _armor_blocks    = new_structure->armor_blocks;
    _eye_blocks      = new_structure->eye_blocks;

    delete new_structure;
}

template<typename T>
int Anatomy::get_map_size(boost::unordered_map<int, boost::unordered_map<int, T>> map) {
    int total_size = 0;
    for (auto & xmap: map) {
        total_size += xmap.second.size();
    }
    return total_size;
}
