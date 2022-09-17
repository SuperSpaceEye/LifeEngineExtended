// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"
#include "LegacyAnatomyMutationLogic.h"

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

Anatomy::Anatomy(const Anatomy & anatomy) {
    _organism_blocks = std::vector(anatomy._organism_blocks);

    _producing_space = std::vector(anatomy._producing_space);
    _eating_space    = std::vector(anatomy._eating_space);
    _killing_space   = std::vector(anatomy._killing_space);

    _mouth_blocks    = anatomy._mouth_blocks;
    _producer_blocks = anatomy._producer_blocks;
    _mover_blocks    = anatomy._mover_blocks;
    _killer_blocks   = anatomy._killer_blocks;
    _armor_blocks    = anatomy._armor_blocks;
    _eye_blocks      = anatomy._eye_blocks;
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
        case BlockTypes::MouthBlock:    mouth_blocks++    ; break;
        case BlockTypes::ProducerBlock: producer_blocks++ ; break;
        case BlockTypes::MoverBlock:    mover_blocks++    ; break;
        case BlockTypes::KillerBlock:   killer_blocks++   ; break;
        case BlockTypes::ArmorBlock:    armor_blocks++    ; break;
        case BlockTypes::EyeBlock:      eye_blocks++      ; break;
        case BlockTypes::EmptyBlock:
        case BlockTypes::FoodBlock:
        case BlockTypes::WallBlock:
            break;
    }

    LegacyAnatomyMutationLogic::create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    LegacyAnatomyMutationLogic::create_eating_space(organism_blocks, eating_space, single_adjacent_space, mouth_blocks);
    LegacyAnatomyMutationLogic::create_killing_space(organism_blocks, killing_space, single_adjacent_space, killer_blocks);

    return LegacyAnatomyMutationLogic::serialize(organism_blocks,
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

    LegacyAnatomyMutationLogic::create_single_adjacent_space(organism_blocks, single_adjacent_space);
    LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

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
        case BlockTypes::MouthBlock:    mouth_blocks--    ; break;
        case BlockTypes::ProducerBlock: producer_blocks-- ; break;
        case BlockTypes::MoverBlock:    mover_blocks--    ; break;
        case BlockTypes::KillerBlock:   killer_blocks--   ; break;
        case BlockTypes::ArmorBlock:    armor_blocks--    ; break;
        case BlockTypes::EyeBlock:      eye_blocks--      ; break;
        case BlockTypes::EmptyBlock:
        case BlockTypes::FoodBlock:
        case BlockTypes::WallBlock:
            break;
    }

    switch (type) {
        case BlockTypes::MouthBlock:    mouth_blocks++    ; break;
        case BlockTypes::ProducerBlock: producer_blocks++ ; break;
        case BlockTypes::MoverBlock:    mover_blocks++    ; break;
        case BlockTypes::KillerBlock:   killer_blocks++   ; break;
        case BlockTypes::ArmorBlock:    armor_blocks++    ; break;
        case BlockTypes::EyeBlock:      eye_blocks++      ; break;
        case BlockTypes::EmptyBlock:
        case BlockTypes::FoodBlock:
        case BlockTypes::WallBlock:
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

    LegacyAnatomyMutationLogic::create_single_adjacent_space(organism_blocks, single_adjacent_space);
    LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    LegacyAnatomyMutationLogic::create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    LegacyAnatomyMutationLogic::create_eating_space(organism_blocks, eating_space, single_adjacent_space, mouth_blocks);
    LegacyAnatomyMutationLogic::create_killing_space(organism_blocks, killing_space, single_adjacent_space, killer_blocks);

    return LegacyAnatomyMutationLogic::serialize(organism_blocks,
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
        case BlockTypes::MouthBlock:    mouth_blocks--    ; break;
        case BlockTypes::ProducerBlock: producer_blocks-- ; break;
        case BlockTypes::MoverBlock:    mover_blocks--    ; break;
        case BlockTypes::KillerBlock:   killer_blocks--   ; break;
        case BlockTypes::ArmorBlock:    armor_blocks--    ; break;
        case BlockTypes::EyeBlock:      eye_blocks--      ; break;
        case BlockTypes::EmptyBlock:
        case BlockTypes::FoodBlock:
        case BlockTypes::WallBlock:
            break;
    }

    int x = _organism_blocks[block_choice].relative_x;
    int y = _organism_blocks[block_choice].relative_y;

    if (x == 0 && y == 0) {
        //if center block is deleted, we need to recenter the anatomy
        LegacyAnatomyMutationLogic::reset_organism_center(_organism_blocks, organism_blocks, x, y);
    } else {
        for (auto &block: _organism_blocks) {
            organism_blocks[block.relative_x][block.relative_y] = BaseGridBlock{block.type, block.rotation};
//            organism_blocks[block.relative_x][block.relative_y].type     = block.type;
//            organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
        }
    }

    organism_blocks[x].erase(organism_blocks[x].find(y));

    LegacyAnatomyMutationLogic::create_single_adjacent_space(organism_blocks, single_adjacent_space);
    LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    LegacyAnatomyMutationLogic::create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, producer_blocks);
    LegacyAnatomyMutationLogic::create_eating_space(organism_blocks, eating_space, single_adjacent_space, mouth_blocks);
    LegacyAnatomyMutationLogic::create_killing_space(organism_blocks, killing_space, single_adjacent_space, killer_blocks);

    return LegacyAnatomyMutationLogic::serialize(organism_blocks,
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

void Anatomy::set_block(BlockTypes type, Rotation rotation, int x, int y) {
    int num_block = 0;
    for (auto & item: _organism_blocks) {
        if (item.relative_x == x && item.relative_y == y) {
            //If delete block, then the decrementing logic will be configured by remove_block
            if (type  != BlockTypes::EmptyBlock) {
                switch (item.type) {
                    case BlockTypes::MouthBlock:    _mouth_blocks--    ; break;
                    case BlockTypes::ProducerBlock: _producer_blocks-- ; break;
                    case BlockTypes::MoverBlock:    _mover_blocks--    ; break;
                    case BlockTypes::KillerBlock:   _killer_blocks--   ; break;
                    case BlockTypes::ArmorBlock:    _armor_blocks--    ; break;
                    case BlockTypes::EyeBlock:      _eye_blocks--      ; break;
                    case BlockTypes::EmptyBlock:
                    case BlockTypes::FoodBlock:
                    case BlockTypes::WallBlock:
                        break;
                }
            }
            break;
        }
        num_block++;
    }

    //if tried to delete an empty space
    if (type == BlockTypes::EmptyBlock && num_block == _organism_blocks.size()) { return;}

    SerializedOrganismStructureContainer *new_structure;

    if (type == BlockTypes::EmptyBlock) {
        new_structure = remove_block(num_block);
    } else {
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> organism_blocks;

        for (auto& block: _organism_blocks) {
            organism_blocks[block.relative_x][block.relative_y].type     = block.type;
            organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
        }

        boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
        boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

        LegacyAnatomyMutationLogic::create_single_adjacent_space(organism_blocks, single_adjacent_space);
        LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

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
         new_structure = add_block(type, -1, rotation, x, y,
                                        organism_blocks, _single_adjacent_space,
                                        _single_diagonal_adjacent_space, single_adjacent_space);
    }

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

    _mouth_blocks    = 0;
    _producer_blocks = 0;
    _mover_blocks    = 0;
    _killer_blocks   = 0;
    _armor_blocks    = 0;
    _eye_blocks      = 0;

    for (auto & block: blocks) {
        switch (block.type) {
            case BlockTypes::MouthBlock:    _mouth_blocks++    ; break;
            case BlockTypes::ProducerBlock: _producer_blocks++ ; break;
            case BlockTypes::MoverBlock:    _mover_blocks++    ; break;
            case BlockTypes::KillerBlock:   _killer_blocks++   ; break;
            case BlockTypes::ArmorBlock:    _armor_blocks++    ; break;
            case BlockTypes::EyeBlock:      _eye_blocks++      ; break;
            default: break;
        }

        organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    LegacyAnatomyMutationLogic::create_single_adjacent_space(organism_blocks, single_adjacent_space);
    LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    LegacyAnatomyMutationLogic::create_producing_space(organism_blocks, producing_space, single_adjacent_space, num_producing_space, _producer_blocks);
    LegacyAnatomyMutationLogic::create_eating_space(organism_blocks, eating_space, single_adjacent_space, _mouth_blocks);
    LegacyAnatomyMutationLogic::create_killing_space(organism_blocks, killing_space, single_adjacent_space, _killer_blocks);

    auto *new_structure = LegacyAnatomyMutationLogic::serialize(organism_blocks,
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

Anatomy &Anatomy::operator=(Anatomy &&other_anatomy) noexcept {
    _mouth_blocks    = other_anatomy._mouth_blocks;
    _producer_blocks = other_anatomy._producer_blocks;
    _mover_blocks    = other_anatomy._mover_blocks;
    _killer_blocks   = other_anatomy._killer_blocks;
    _armor_blocks    = other_anatomy._armor_blocks;
    _eye_blocks      = other_anatomy._eye_blocks;

    _organism_blocks = std::move(other_anatomy._organism_blocks);
    _producing_space = std::move(other_anatomy._producing_space);
    _eating_space    = std::move(other_anatomy._eating_space);
    _killing_space   = std::move(other_anatomy._killing_space);

    return *this;
}
