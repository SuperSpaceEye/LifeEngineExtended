//
// Created by spaceeye on 15.09.22.
//

#include "Anatomy.h"
#include "LegacyAnatomyMutationLogic.h"

void LegacyAnatomyMutationLogic::set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                                     boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                                     boost::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                                                     const BaseGridBlock &block) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset)) {
        single_adjacent_space[x + x_offset][y + y_offset] = true;
    }
}

void LegacyAnatomyMutationLogic::create_single_adjacent_space(
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

void LegacyAnatomyMutationLogic::set_single_diagonal_adjacent(int x, int y, int x_offset, int y_offset,
                                                              boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                                              boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                                              boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space) {
    if (!organism_blocks[x+x_offset].count(y+y_offset)       &&
        !single_adjacent_space[x+x_offset].count(y+y_offset) &&
        !single_diagonal_adjacent_space[x+x_offset].count(y+y_offset))
    {single_diagonal_adjacent_space[x+x_offset][y+y_offset] = true;}
}

void LegacyAnatomyMutationLogic::create_single_diagonal_adjacent_space(
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

void LegacyAnatomyMutationLogic::create_producing_space(
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

void LegacyAnatomyMutationLogic::create_eating_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
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

void LegacyAnatomyMutationLogic::create_killing_space(boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
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

SerializedOrganismStructureContainer * LegacyAnatomyMutationLogic::serialize(
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
    std::vector<SerializedOrganismBlockContainer> _eye_blocks_vector;

    _organism_blocks.reserve(get_map_size(organism_blocks));

    _producing_space.reserve(producer_blocks);
    _eating_space.reserve(   get_map_size(eating_space));
    _killing_space.reserve(get_map_size(killing_space));

    serialize_organism_blocks(organism_blocks, _organism_blocks);

    serialize_producing_space(producing_space, num_producing_space, _producing_space);

    serialize_eating_space(eating_space, _eating_space);

    serialize_killing_space(killing_space, _killing_space);

    serialize_eye_blocks(_organism_blocks, _eye_blocks_vector, eye_blocks);

    return new SerializedOrganismStructureContainer{_organism_blocks,

                                                    _producing_space,
                                                    _eating_space,
                                                    _killing_space,
                                                    _eye_blocks_vector,

                                                    mouth_blocks,
                                                    producer_blocks,
                                                    mover_blocks,
                                                    killer_blocks,
                                                    armor_blocks,
                                                    eye_blocks};
}

void
LegacyAnatomyMutationLogic::serialize_killing_space(const boost::unordered_map<int, boost::unordered_map<int, bool>> &killing_space,
                                                    std::vector<SerializedAdjacentSpaceContainer> &_killing_space) {
    for (auto const &xmap: killing_space) {
        for (auto const &yxmap: xmap.second) {
            _killing_space.emplace_back(xmap.first, yxmap.first);
        }
    }
}

void
LegacyAnatomyMutationLogic::serialize_eating_space(const boost::unordered_map<int, boost::unordered_map<int, bool>> &eating_space,
                                                   std::vector<SerializedAdjacentSpaceContainer> &_eating_space) {
    for (auto const &xmap: eating_space) {
        for (auto const &yxmap: xmap.second) {
            _eating_space.emplace_back(xmap.first, yxmap.first);
        }
    }
}

void LegacyAnatomyMutationLogic::serialize_organism_blocks(
        const boost::unordered_map<int, boost::unordered_map<int, BaseGridBlock>> &organism_blocks,
        std::vector<SerializedOrganismBlockContainer> &_organism_blocks) {//item.first = position in map, item.second = content
    for (auto const &xmap: organism_blocks) {
        for (auto const &yxmap: xmap.second) {
            _organism_blocks.emplace_back(yxmap.second.type, yxmap.second.rotation, xmap.first, yxmap.first);
        }
    }
}

void LegacyAnatomyMutationLogic::serialize_producing_space(
        const boost::unordered_map<int, boost::unordered_map<int, ProducerAdjacent>> &producing_space,
        const std::vector<int> &num_producing_space,
        std::vector<std::vector<SerializedAdjacentSpaceContainer>> &_producing_space) {

    for (int i = 0; i < num_producing_space.size(); i++) {
        _producing_space.emplace_back(std::vector<SerializedAdjacentSpaceContainer>());
        _producing_space[i].reserve(num_producing_space[i]);
    }

    for (auto const &xmap: producing_space) {
        for (auto const &yxmap: xmap.second) {
            _producing_space[yxmap.second.producer].emplace_back(SerializedAdjacentSpaceContainer(xmap.first, yxmap.first));
        }
    }

    //pruning empty producing spaces
    int deleted = 0;
    auto size = _producing_space.size();
    for (int i = 0; i < size; i++) {
        if (_producing_space[i-deleted].empty()) {
            _producing_space.erase(_producing_space.begin() + i - deleted);
            deleted++;
        }
    }

    //TODO downsizing vector?
//    std::vector<std::vector<SerializedAdjacentSpaceContainer>>(_producing_space).swap(_producing_space);
}

void
LegacyAnatomyMutationLogic::serialize_eye_blocks(const std::vector<SerializedOrganismBlockContainer> &organism_blocks,
                                                 std::vector<SerializedOrganismBlockContainer> &eye_blocks_vector,
                                                 int eye_blocks) {
    eye_blocks_vector.reserve(eye_blocks);
    for (const auto & block: organism_blocks) {
        if (block.type == BlockTypes::EyeBlock) {
            eye_blocks_vector.emplace_back(block);
        }
    }
}

template<typename T>
int LegacyAnatomyMutationLogic::get_map_size(boost::unordered_map<int, boost::unordered_map<int, T>> map) {
    int total_size = 0;
    for (auto & xmap: map) {
        total_size += xmap.second.size();
    }
    return total_size;
}