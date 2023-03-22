// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#include "Anatomy.h"

using BT = BlockTypes;

Anatomy::Anatomy(SerializedOrganismStructureContainer *structure) {
    move_s(structure);
    delete structure;
}

Anatomy::Anatomy(const Anatomy & anatomy) {
    copy_s(&anatomy);
}

SerializedOrganismStructureContainer *
Anatomy::make_container(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &_organism_blocks,
                        boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                        AnatomyCounters<int, NUM_ORGANISM_BLOCKS, (std::string_view *) SW_ORGANISM_BLOCK_NAMES> &_c) const {
    std::vector<int> num_producing_space;

    boost::unordered_map<int, boost::unordered_map<int, std::vector<int>>> _producing_space;
    boost::unordered_map<int, boost::unordered_map<int, int>> _eating_space;
    boost::unordered_map<int, boost::unordered_map<int, int>> _killing_space;

    SimpleAnatomyMutationLogic::create_producing_space(_organism_blocks, _producing_space, single_adjacent_space, num_producing_space, _c[BT::ProducerBlock]);
    SimpleAnatomyMutationLogic::create_eating_space(_organism_blocks, _eating_space, single_adjacent_space, _c[BT::MouthBlock]);
    SimpleAnatomyMutationLogic::create_killing_space(_organism_blocks, _killing_space, single_adjacent_space, _c[BT::KillerBlock]);

    return SimpleAnatomyMutationLogic::serialize(_organism_blocks,
                                                 _producing_space,
                                                 _eating_space,
                                                 _killing_space,
                                                 num_producing_space,
                                                 _c);
}

SerializedOrganismStructureContainer * Anatomy::add_block(BlockTypes type, int block_choice, Rotation rotation, int x_,
                                                          int y_,
                                                          boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &_organism_blocks,
                                                          std::vector<SerializedAdjacentSpaceContainer> &_single_adjacent_space,
                                                          std::vector<SerializedAdjacentSpaceContainer> &_single_diagonal_adjacent_space,
                                                          boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space) {
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
    _organism_blocks[x][y] = block;

    auto _c = make_map();
    _c = c;
    _c[type]++;

    return make_container(_organism_blocks, single_adjacent_space, _c);
}

SerializedOrganismStructureContainer * Anatomy::add_random_block(OrganismBlockParameters& bp, lehmer64 &mt) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> _organism_blocks;

    for (auto& block: organism_blocks) {
        _organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        _organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    SimpleAnatomyMutationLogic::create_single_adjacent_space(_organism_blocks, single_adjacent_space);
    SimpleAnatomyMutationLogic::create_single_diagonal_adjacent_space(_organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

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

    float total_chance = [&](){float summ=0;for(auto&item:bp.pa){summ+=item.chance_weight;}return summ;}();

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(mt);
    int   block_choice = std::uniform_int_distribution<int>{0, int(_single_adjacent_space.size()+_single_diagonal_adjacent_space.size())-1}(mt);

    for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
        auto & item = bp.pa[i];
        if (type_choice < item.chance_weight) {
            return add_block((BlockTypes)(i+1),
                             block_choice, static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(mt)),
                             0, 0, _organism_blocks, _single_adjacent_space,
                             _single_diagonal_adjacent_space, single_adjacent_space);
        }
        type_choice -= item.chance_weight;
    }

    // if for some reason the choice fails.
    return add_block((BlockTypes)NUM_ORGANISM_BLOCKS, block_choice, static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(mt)),
                     0, 0, _organism_blocks, _single_adjacent_space,
                     _single_diagonal_adjacent_space, single_adjacent_space);
}

SerializedOrganismStructureContainer * Anatomy::change_block(BlockTypes type, int block_choice, lehmer64 *mt) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> _organism_blocks;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    auto _c = make_map();
    _c = c;

    _c[organism_blocks[block_choice].type]--;
    _c[type]++;

    for (int i = 0; i < organism_blocks.size(); i ++) {
        _organism_blocks[organism_blocks[i].relative_x][organism_blocks[i].relative_y].type      = organism_blocks[i].type;
        _organism_blocks[organism_blocks[i].relative_x][organism_blocks[i].relative_y].rotation  = organism_blocks[i].rotation;
        if (i == block_choice) {
            _organism_blocks[organism_blocks[i].relative_x][organism_blocks[i].relative_y].type = type;
            if (type == BlockTypes::EyeBlock) {
                _organism_blocks[organism_blocks[i].relative_x][organism_blocks[i].relative_y].rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
            }
        }
    }

    SimpleAnatomyMutationLogic::create_single_adjacent_space(_organism_blocks, single_adjacent_space);
//    SimpleAnatomyMutationLogic::create_single_diagonal_adjacent_space(_organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    return make_container(_organism_blocks, single_adjacent_space, _c);
}

SerializedOrganismStructureContainer * Anatomy::change_random_block(OrganismBlockParameters& bp, lehmer64 &gen) {
    float total_chance = [&](){float summ=0;for(auto&item:bp.pa){summ+=item.chance_weight;}return summ;}();

    float type_choice  = std::uniform_real_distribution<float>{0, total_chance}(gen);
    int   block_choice = std::uniform_int_distribution<int>{0, int(organism_blocks.size()) - 1}(gen);

    for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
        auto &item = bp.pa[i];
        if (type_choice < item.chance_weight) {
            return change_block((BlockTypes)(i+1), block_choice, &gen);
        }
        type_choice -= item.chance_weight;
    }

    return change_block((BlockTypes)NUM_ORGANISM_BLOCKS, block_choice, &gen);
}

SerializedOrganismStructureContainer * Anatomy::remove_block(int block_choice) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> _organism_blocks;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    auto _c = make_map();
    _c = c;
    _c[organism_blocks[block_choice].type]--;

    int x = organism_blocks[block_choice].relative_x;
    int y = organism_blocks[block_choice].relative_y;

    for (auto &block: organism_blocks) {
        _organism_blocks[block.relative_x][block.relative_y] = BaseGridBlock{block.type, block.rotation};
    }

    _organism_blocks[x].erase(_organism_blocks[x].find(y));

    SimpleAnatomyMutationLogic::create_single_adjacent_space(_organism_blocks, single_adjacent_space);
//    SimpleAnatomyMutationLogic::create_single_diagonal_adjacent_space(_organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    return make_container(_organism_blocks, single_adjacent_space, _c);
}

SerializedOrganismStructureContainer * Anatomy::remove_random_block(lehmer64 &gen) {
    int block_choice = std::uniform_int_distribution<int>{0, int(organism_blocks.size()) - 1}(gen);
    return remove_block(block_choice);
}

void Anatomy::set_block(BlockTypes type, Rotation rotation, int x, int y) {
//    if (type == BlockTypes::EmptyBlock) { return;}
    int num_block = 0;
    for (auto & item: organism_blocks) {
        if (item.relative_x == x && item.relative_y == y) {
            //If delete block, then the decrementing logic will be configured by remove_block
            if (type  != BlockTypes::EmptyBlock) {
                c[item.type]--;
            }
            break;
        }
        num_block++;
    }

    //if tried to delete an empty space
    if (type == BlockTypes::EmptyBlock && num_block == organism_blocks.size()) { return;}

    SerializedOrganismStructureContainer *new_structure;

    if (type == BlockTypes::EmptyBlock) {
        new_structure = remove_block(num_block);
    } else {
        boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> _organism_blocks;

        for (auto& block: organism_blocks) {
            _organism_blocks[block.relative_x][block.relative_y].type     = block.type;
            _organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
        }

        boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
        boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

        SimpleAnatomyMutationLogic::create_single_adjacent_space(_organism_blocks, single_adjacent_space);
        SimpleAnatomyMutationLogic::create_single_diagonal_adjacent_space(_organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

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
                                   _organism_blocks, _single_adjacent_space,
                                   _single_diagonal_adjacent_space, single_adjacent_space);
    }

    move_s(new_structure);
    delete new_structure;
}

void Anatomy::set_many_blocks(const std::vector<SerializedOrganismBlockContainer> &blocks) {
    boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> _organism_blocks;

    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;

    auto _c = make_map();
    for (auto & block: blocks) {
        _c[block.type]++;

        _organism_blocks[block.relative_x][block.relative_y].type     = block.type;
        _organism_blocks[block.relative_x][block.relative_y].rotation = block.rotation;
    }

    SimpleAnatomyMutationLogic::create_single_adjacent_space(_organism_blocks, single_adjacent_space);
//    SimpleAnatomyMutationLogic::create_single_diagonal_adjacent_space(_organism_blocks, single_adjacent_space, single_diagonal_adjacent_space);

    auto *new_structure = make_container(_organism_blocks, single_adjacent_space, _c);
    move_s(new_structure);
    delete new_structure;
}

Anatomy &Anatomy::operator=(Anatomy &&other_anatomy) noexcept {move_s(&other_anatomy);return *this;}
Anatomy &Anatomy::operator=(const Anatomy &other_anatomy)     {copy_s(&other_anatomy);return *this;}

//TODO
Vector2<int> Anatomy::recenter_blocks(bool imaginary_center) {
    if (imaginary_center) {
        return recenter_to_imaginary();
    } else {
        return recenter_to_existing();
    }
}

void Anatomy::subtract_difference(int x, int y) {
    for (auto & block: organism_blocks) {
        block.relative_x -= x;
        block.relative_y -= y;
    }

    for (auto & spe: producing_space) {
        for (auto & pc: spe) {
            pc.relative_x -= x;
            pc.relative_y -= y;
        }
    }

    for (auto & block: eating_space) {
        block.relative_x -= x;
        block.relative_y -= y;
    }

    for (auto & block: killing_space) {
        block.relative_x -= x;
        block.relative_y -= y;
    }

    for (auto & block: eye_block_vec) {
        block.relative_x -= x;
        block.relative_y -= y;
    }
}

Vector2<int> Anatomy::recenter_to_existing() {
    if (organism_blocks.empty()) { return {0, 0};}
    int block_pos_in_vec = 0;
    //the position of a block will definitely never be bigger than this.
    Vector2<int32_t> abs_pos{INT32_MAX/4, INT32_MAX/4};

    //will find the closest cell to the center
    int i = 0;
    for (auto & block: organism_blocks) {
        if (std::abs(block.relative_x) + std::abs(block.relative_y) < abs_pos.x + abs_pos.y) {
            abs_pos = {std::abs(block.relative_x), std::abs(block.relative_y)};
            block_pos_in_vec = i;
        }
        i++;
    }

    Vector2<int32_t> new_center_pos = {organism_blocks[block_pos_in_vec].relative_x, organism_blocks[block_pos_in_vec].relative_y};

    subtract_difference(new_center_pos.x, new_center_pos.y);
    return {new_center_pos.x, new_center_pos.y};
}

Vector2<int> Anatomy::recenter_to_imaginary() {
    if (organism_blocks.empty()) { return {0, 0};}
    Vector2 min{0, 0};
    Vector2 max{0, 0};

    for (auto & block: organism_blocks) {
        if (block.relative_x < min.x) {min.x = block.relative_x;}
        if (block.relative_y < min.y) {min.y = block.relative_y;}
        if (block.relative_x > max.x) {max.x = block.relative_x;}
        if (block.relative_y > max.y) {max.y = block.relative_y;}
    }

    auto diff_x = (max.x - min.x) / 2 + min.x;
    auto diff_y = (max.y - min.y) / 2 + min.y;

    subtract_difference(diff_x, diff_y);

    return {diff_x, diff_y};
}