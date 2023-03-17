// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 15.09.22.
//

#include "Organism/Anatomy/AnatomyContainers.h"

#include "OrganismConstructionCode.h"

using BT = BlockTypes;

//Will simulate changes in cursor position and find max/min positions that occ will need to actually construct anatomy
std::array<int, 4> OrganismConstructionCode::calculate_construction_edges() {
    // min x, max x, min y, max y
    auto edges = std::array<int, 4>({0, 0, 0, 0});

    auto shift_values = std::array<std::array<int, 2>, 8> {
        std::array<int, 2>{ 0,-1},
        std::array<int, 2>{-1,-1},
        std::array<int, 2>{-1, 0},
        std::array<int, 2>{-1, 1},
        std::array<int, 2>{ 0, 1},
        std::array<int, 2>{ 1, 1},
        std::array<int, 2>{ 1, 0},
        std::array<int, 2>{ 1,-1},
    };
    //position of cursor
    int x = 0;
    int y = 0;

    //changeable origin of cursor
    int origin_x = 0;
    int origin_y = 0;

    for (int i = 0; i < occ_vector.size(); i++) {
        auto instruction = occ_vector[i];
        bool last_instruction = i == occ_vector.size()-1;

        if ((
                int(instruction) >= int(OCCInstruction::SetUnderBlockMouth)
             && int(instruction)  < int(OCCInstruction::SetUnderBlockMouth)+NUM_ORGANISM_BLOCKS)
             || (
                int(instruction) >= int(OCCInstruction::SetBlockMouth)
             && int(instruction)  < int(OCCInstruction::SetBlockMouth)+NUM_ORGANISM_BLOCKS
                  )) {
            if (x < edges[0]) { edges[0] = x;}
            if (x > edges[1]) { edges[1] = x;}
            if (y < edges[2]) { edges[2] = y;}
            if (y > edges[3]) { edges[3] = y;}
        }

        if (int(instruction) >= int(OCCInstruction::ShiftUp)
         && int(instruction) <= int(OCCInstruction::ShiftRightUp)) {
            auto shift = shift_values[static_cast<int>(instruction)];

            if (!last_instruction) {
                auto next_instruction = occ_vector[i + 1];

                if (int(next_instruction) >= int(OCCInstruction::SetBlockMouth)
                 && int(next_instruction) < int(OCCInstruction::SetBlockMouth)+NUM_ORGANISM_BLOCKS) {
                    if (x + shift[0] < edges[0]) { edges[0] = x + shift[0];}
                    if (x + shift[0] > edges[1]) { edges[1] = x + shift[0];}
                    if (y + shift[1] < edges[2]) { edges[2] = y + shift[1];}
                    if (y + shift[1] > edges[3]) { edges[3] = y + shift[1];}
                    i+=1;
                    continue;
                }
            }
            x += shift[0];
            y += shift[1];
        }

        if (instruction == OCCInstruction::ResetToOrigin) {
            x = origin_x;
            y = origin_y;
            continue;
        }

        if (instruction == OCCInstruction::SetOrigin) {
            origin_x = x;
            origin_y = y;
            continue;
        }
    }

    return edges;
}

SerializedOrganismStructureContainer *OrganismConstructionCode::compile_code(OCCLogicContainer &occ_c,
                                                                             bool organisms_grow) {
    auto * container = new SerializedOrganismStructureContainer();

    auto edges = calculate_construction_edges();
    auto edge_width = std::abs(edges[0]) + std::abs(edges[1]) + 1;
    auto edge_height = std::abs(edges[2]) + std::abs(edges[3]) + 1;
    if (edge_width > occ_c.occ_width || edge_height > occ_c.occ_height) {
        occ_c.occ_width  = edge_width;
        occ_c.occ_height = edge_height;
        occ_c.occ_main_block_construction_space.resize(edge_width*edge_height);
        //TODO i don't want to bother with calculating additional values right now.
        occ_c.occ_eating_space   .resize((edge_width + 2) * (edge_height + 2));
        occ_c.occ_killing_space  .resize((edge_width + 2) * (edge_height + 2));
    }

    container->organism_blocks = compile_base_structure(container, occ_c, edges);
    edges[0]--;edges[1]++;edges[2]--;edges[3]++;

    return compile_spaces(occ_c, edges, container->organism_blocks, container, organisms_grow);
}

void set_block(int x, int y, BlockTypes type, Rotation rotation, OCCLogicContainer &occ_c,
               std::vector<SerializedOrganismBlockContainer> &temp_blocks,
               SerializedOrganismStructureContainer *structure_container, int center_x, int center_y) {
#ifdef __DEBUG__
    if (x < 0 || y < 0 || x >= occ_c.occ_width || y >= occ_c.occ_height) {
        throw std::runtime_error("");
    }
    if ((int)rotation >= 4) {throw std::runtime_error("");}
#endif
    auto & block = occ_c.occ_main_block_construction_space[x + y * occ_c.occ_width];
    bool set_block = false;

    //If block got from space was not placed for current anatomy, then treat it as empty space, otherwise find the actual
    //position in temp_blocks and change it.
    if (block.counter < occ_c.main_counter) {
        set_block = true;
    }

    if (set_block) {
        temp_blocks.emplace_back(type, rotation, x - center_x, y - center_y);
        block.counter = occ_c.main_counter;
        block.parent_block_pos = temp_blocks.size()-1;
        block.type = type;
    } else {
        structure_container->c[block.type]--;
        temp_blocks[block.parent_block_pos].type = type;
        temp_blocks[block.parent_block_pos].rotation = rotation;
        block.type = type;
    }

    structure_container->c[type]++;
}

void set_rotation(int x, int y, Rotation rotation, OCCLogicContainer &occ_c,
                  std::vector<SerializedOrganismBlockContainer> &temp_blocks) {
#ifdef __DEBUG__
    if ((int)rotation >= 4) {throw std::runtime_error("");}
#endif
    if (x < 0 || y < 0 || x >= occ_c.occ_width || y >= occ_c.occ_height) {return;}
    auto & block = occ_c.occ_main_block_construction_space[x + y * occ_c.occ_width];
    if (block.counter == occ_c.main_counter) {
        temp_blocks[block.parent_block_pos].rotation = rotation;
    }
}

std::vector<SerializedOrganismBlockContainer>
OrganismConstructionCode::compile_base_structure(SerializedOrganismStructureContainer *container,
                                                 OCCLogicContainer &occ_c, const std::array<int, 4> &edges) {
    std::vector<SerializedOrganismBlockContainer> blocks;
    auto shift_values = std::array<std::array<int, 2>, 8> {
            std::array<int, 2>{ 0,-1},
            std::array<int, 2>{-1,-1},
            std::array<int, 2>{-1, 0},
            std::array<int, 2>{-1, 1},
            std::array<int, 2>{ 0, 1},
            std::array<int, 2>{ 1, 1},
            std::array<int, 2>{ 1, 0},
            std::array<int, 2>{ 1,-1},
    };
    auto shift = shift_values[0];

    Rotation base_rotation = Rotation::UP;
    int cursor_x = std::abs(edges[0]);
    int cursor_y = std::abs(edges[2]);

    //actual center
    int center_x = cursor_x;
    int center_y = cursor_y;

    int origin_x = cursor_x;
    int origin_y = cursor_y;

    occ_c.main_counter++;

    for (int i = 0; i < occ_vector.size(); i++) {
        auto instruction = occ_vector[i];
        OCCInstruction next_instruction;
        //if not last instruction
        //is needed for shift instruction logic.
        if (i != occ_vector.size()-1) {
            next_instruction = occ_vector[i+1];
        } else {
            next_instruction = OCCInstruction::ResetToOrigin;
        }

        if (int(instruction) >= int(OCCInstruction::ShiftUp)
            && int(instruction) <= int(OCCInstruction::ShiftRightUp)) {
            bool pass = false;
            shift_instruction_part(container, occ_c, shift_values, shift, base_rotation, cursor_x, cursor_y,
                                   center_x, center_y,
                                   instruction, next_instruction, blocks, i, pass);
            if (pass) { continue; }
            shift = shift_values[static_cast<int>(instruction)];
            cursor_x += shift[0];
            cursor_y += shift[1];
            continue;
        }

        if (int(instruction) >= int(OCCInstruction::ApplyRotationUp)
         && int(instruction) <= int(OCCInstruction::ApplyRotationRight)) {
            base_rotation = (Rotation)(int(instruction)-int(OCCInstruction::ApplyRotationUp));
            continue;
        }

        if (int(instruction) >= int(OCCInstruction::SetRotationUp)
         && int(instruction) <= int(OCCInstruction::SetRotationRight)) {
            set_rotation(cursor_x, cursor_y, (Rotation)(int(instruction)-int(OCCInstruction::SetRotationUp)), occ_c, blocks);
            continue;
        }

        if (instruction == OCCInstruction::ResetToOrigin) {
            cursor_x = origin_x;
            cursor_y = origin_y;
            continue;
        }

        if (instruction == OCCInstruction::SetOrigin) {
            origin_x = cursor_x;
            origin_y = cursor_y;
            continue;
        }

        if (int(instruction) >= int(OCCInstruction::SetBlockMouth)
         && int(instruction) < int(OCCInstruction::SetBlockMouth) + NUM_ORGANISM_BLOCKS) {
            set_block(cursor_x, cursor_y, (BlockTypes)(int(instruction)-int(OCCInstruction::SetBlockMouth)+1), base_rotation, occ_c, blocks, container, center_x, center_y);
            continue;
        }

        if (int(instruction) >= int(OCCInstruction::SetUnderBlockMouth)
            && int(instruction) < int(OCCInstruction::SetUnderBlockMouth) + NUM_ORGANISM_BLOCKS) {
            set_block(cursor_x, cursor_y, (BlockTypes)(int(instruction)-int(OCCInstruction::SetUnderBlockMouth)+1), base_rotation, occ_c, blocks, container, center_x, center_y);
            continue;
        }
    }

    return blocks;
}

void OrganismConstructionCode::shift_instruction_part(SerializedOrganismStructureContainer *container,
                                                      OCCLogicContainer &occ_c,
                                                      const std::array<std::array<int, 2>, 8> &shift_values,
                                                      std::array<int, 2> &shift, const Rotation base_rotation, int cursor_x,
                                                      int cursor_y, int center_x, int center_y,
                                                      const OCCInstruction instruction,
                                                      const OCCInstruction next_instruction,
                                                      std::vector<SerializedOrganismBlockContainer> &blocks, int &i,
                                                      bool &pass) {
    if (int(next_instruction) >= int(OCCInstruction::SetBlockMouth)
        && int(next_instruction) < int(OCCInstruction::SetBlockMouth) + NUM_ORGANISM_BLOCKS) {
        shift = shift_values[static_cast<int>(instruction)];
        set_block(cursor_x + shift[0], cursor_y + shift[1],
                //set block instruction are in the same order as block types.
                  static_cast<BlockTypes>(
                          static_cast<int>(next_instruction)- NON_SET_BLOCK_OCC_INSTRUCTIONS + 1),
                  base_rotation, occ_c, blocks, container, center_x, center_y);
        i++;
        pass = true;
    }
}

//Will compile spaces all at the same time in one go.
SerializedOrganismStructureContainer *
OrganismConstructionCode::compile_spaces(OCCLogicContainer &occ_c, const std::array<int, 4> &edges,
                                         std::vector<SerializedOrganismBlockContainer> &organism_blocks,
                                         SerializedOrganismStructureContainer *container, bool organisms_grow) {
    constexpr auto shifting_positions = std::array<std::array<int, 2>, 4> {
        std::array<int, 2>{ 0,-1},
        std::array<int, 2>{-1, 0},
        std::array<int, 2>{ 0, 1},
        std::array<int, 2>{ 1, 0},
    };

    occ_c.spaces_counter++;

    auto &producing_space = container->producing_space;
    auto &eating_space    = container->eating_space;
    auto &killer_space    = container->killing_space;
    auto &eye_blocks_vec  = container->eye_block_vec;

    auto & killer_mask = container->killer_mask; killer_mask.reserve(container->c[BT::MouthBlock]);
    auto & eating_mask = container->eating_mask; eating_mask.reserve(container->c[BT::KillerBlock]);

    eye_blocks_vec.reserve(container->c[BT::EyeBlock]);

    auto temp_producing_space = std::vector<OCCSerializedProducingSpace>();

    producing_space.resize(container->c[BT::ProducerBlock]);
    //TODO probably not very space efficient
    eating_space.reserve(container->c[BT::MouthBlock]*4);
    killer_space.reserve(container->c[BT::KillerBlock]*4);
    uint32_t eating_c = 0;
    uint32_t killer_c = 0;

    int producer = -1;

    auto center_x = std::abs(edges[0]);
    auto center_y = std::abs(edges[2]);

    for (auto & block: organism_blocks) {
        auto x = block.relative_x + center_x;
        auto y = block.relative_y + center_y;

        switch (block.type) {
            case BlockTypes::ProducerBlock:
                producer++;
                for (auto & shift: shifting_positions) {
                    int x_ = x + shift[0];
                    int y_ = y + shift[1];

                    //organism blocks can only affect area directly connected to them, so if the space is beyond computed
                    // edges, then no other block can affect the same area
                    if (x_ == 0 || y_ == 0 || x_ >= occ_c.occ_width || y_ >= occ_c.occ_height) {
                        temp_producing_space.emplace_back(producer, x_-center_x, y_-center_y);
                        continue;
                    }
                    //if space position is inside an organism block
                    if (occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_width].counter == occ_c.main_counter) {
                        continue;
                    }
                    //we do not care if there are other producing spaces overlapping this one
                    temp_producing_space.emplace_back(producer, x_-center_x, y_-center_y);
                }
                break;
            case BlockTypes::MouthBlock:
                for (auto & shift: shifting_positions) {
                    int x_ = x + shift[0];
                    int y_ = y + shift[1];

                    //organism blocks can only affect area directly connected to them, so if the space is beyond computed
                    // edges, then no other block can affect the same area
                    if (x_ == 0 || y_ == 0 || x_ >= occ_c.occ_width || y_ >= occ_c.occ_height) {
                        eating_space.emplace_back(x_-center_x, y_-center_y);
                        eating_c++;
                        continue;
                    }
                    //if space position is inside an organism block
                    if (occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_width].counter == occ_c.main_counter) {
                        continue;
                    }
                    // if the space was not initialized before
                    if (occ_c.occ_eating_space[x_ + y_ * (occ_c.occ_width+2)].counter != occ_c.spaces_counter || organisms_grow) {
                        eating_space.emplace_back(x_-center_x, y_-center_y, 1);
                        auto & block = occ_c.occ_eating_space[x_ + y_ * (occ_c.occ_width+2)];
                        block.parent_block_pos = eating_space.size() - 1;
                        block.counter = occ_c.spaces_counter;
                        eating_c++;
                    } else {
                        eating_space[occ_c.occ_eating_space[x_ + y_ * (occ_c.occ_width+2)].parent_block_pos].num++;
                    }
                }
                eating_mask.emplace_back(eating_c);
                break;
            case BlockTypes::KillerBlock:
                for (auto & shift: shifting_positions) {
                    int x_ = x + shift[0];
                    int y_ = y + shift[1];

                    //organism blocks can only affect area directly connected to them, so if the space is beyond computed
                    // edges, then no other block can affect the same area
                    if (x_ == 0 || y_ == 0 || x_ >= occ_c.occ_width || y_ >= occ_c.occ_height) {
                        killer_space.emplace_back(x_-center_x, y_-center_y);
                        killer_c++;
                        continue;
                    }
                    //if space position is inside an organism block
                    if (occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_width].counter == occ_c.main_counter) {
                        continue;
                    }
                    // if the space was not initialized before
                    if (occ_c.occ_killing_space[x_ + y_ * (occ_c.occ_width+2)].counter != occ_c.spaces_counter || organisms_grow) {
                        killer_space.emplace_back(x_-center_x, y_-center_y, 1);
                        auto & block = occ_c.occ_killing_space[x_ + y_ * (occ_c.occ_width+2)];
                        block.parent_block_pos = killer_space.size() - 1;
                        block.counter = occ_c.spaces_counter;
                        killer_c++;
                    } else {
                        killer_space[occ_c.occ_killing_space[x_ + y_ * (occ_c.occ_width+2)].parent_block_pos].num++;
                    }
                }
                killer_mask.emplace_back(killer_c);
                break;
            case BlockTypes::EyeBlock:
#ifdef __DEBUG__
        if ((int)block.rotation < 0 || (int)block.rotation >= 4 ) {throw std::runtime_error("");}
#endif
                eye_blocks_vec.emplace_back(block);
                break;
            default: break;
        }
    }

    for (auto & producing: temp_producing_space) {
        producing_space[producing.producer].emplace_back(producing.x, producing.y);
    }

    //pruning empty producing spaces
    int deleted = 0;
    auto size = producing_space.size();
    for (int i = 0; i < size; i++) {
        if (producing_space[i-deleted].empty()) {
            producing_space.erase(producing_space.begin() + i - deleted);
            deleted++;
        }
    }

#ifdef __DEBUG__
    if (organisms_grow) {
        if (!killer_mask.empty()) {
            assert(killer_mask.back() == killer_space.size());
        } else {assert(killer_space.empty());}

        if (!eating_mask.empty()) {
        assert(eating_mask.back() == eating_space.size());
        } else {assert(eating_space.empty());}

        assert(eating_mask.size() == container->c[BT::MouthBlock]);
        assert(killer_mask.size() == container->c[BT::KillerBlock]);
    }
#endif
    return container;
}

std::vector<OCCInstruction> create_random_group(int group_size, OCCParameters &occp, lehmer64 &gen) {
    std::vector<OCCInstruction> group;
    group.reserve(group_size);

    for (int i = 0; i < group_size; i++) {
        if (occp.uniform_occ_instructions_mutation) {
            group.emplace_back(static_cast<OCCInstruction>(std::uniform_int_distribution<int>(0, NON_SET_BLOCK_OCC_INSTRUCTIONS
                                                                                                +SET_BLOCK_OCC_INSTRUCTIONS-1)(gen)));
        } else {
            group.emplace_back(static_cast<OCCInstruction>(occp.occ_instructions_mutation_discrete_distribution(gen)));
        }
    }
    return group;
}

OrganismConstructionCode OrganismConstructionCode::mutate(OCCParameters &occp, lehmer64 &gen) {
    auto child_code = OrganismConstructionCode(*this);

    OCCMutations mutation_type;
    int group_size;

    if (occp.uniform_mutation_distribution) {
        //5 mutations
        mutation_type = static_cast<OCCMutations>(std::uniform_int_distribution<int>(0, 4)(gen));
    } else {
        mutation_type = static_cast<OCCMutations>(occp.mutation_discrete_distribution(gen));
    }

    if (occp.uniform_group_size_distribution) {
        group_size = std::uniform_int_distribution<int>(1, occp.max_group_size)(gen);
    } else {
        group_size = occp.group_size_discrete_distribution(gen)+1;
    }

    switch (mutation_type) {
        //just append group to the end
        case OCCMutations::AppendRandom: {
            auto group = create_random_group(group_size, occp, gen);
            child_code.occ_vector.insert(child_code.occ_vector.end(), group.begin(), group.end());
        }
        break;
        //insert group into sequence
        case OCCMutations::InsertRandom: {
            auto group = create_random_group(group_size, occp, gen);
            int position = std::uniform_int_distribution<int>(0, child_code.occ_vector.size()-1)(gen);

            child_code.occ_vector.insert(child_code.occ_vector.begin()+position, group.begin(), group.end());
        }
            break;
        //Overwrite existing part of a sequence with a group
        case OCCMutations::ChangeRandom: {
            auto group = create_random_group(group_size, occp, gen);

            int position = std::uniform_int_distribution<int>(0, child_code.occ_vector.size()-1)(gen);

            int iterated = 0;
            auto iterator = child_code.occ_vector.begin()+position;

            while (iterator != child_code.occ_vector.end() && iterated < group.size()) {
                (*iterator) = group[iterated];

                ++iterated;
                ++iterator;
            }
        }
            break;
        //delete part of a sequence with group_size
        case OCCMutations::DeleteRandom: {
            int position = std::uniform_int_distribution<int>(0, child_code.occ_vector.size()-1)(gen);
            int allowed_erasing = std::min<int>(group_size, child_code.occ_vector.size()-position-1);

            auto position_iterator = child_code.occ_vector.begin()+position;
            child_code.occ_vector.erase(position_iterator, position_iterator+allowed_erasing);
        }
            break;
        //swap instructions with size of group_size to the left or right by some distance
        case OCCMutations::SwapRandom: {
            if (child_code.occ_vector.size() == 1) { break;}
            int position = std::uniform_int_distribution<int>(0, child_code.occ_vector.size()-1)(gen);
            int distance;

            if (occp.uniform_swap_distance) {
                distance = std::uniform_int_distribution<int>(0, occp.max_distance)(gen);
            } else {
                distance = occp.swap_distance_mutation_discrete_distribution(gen);
            }

            //will determine whenever the shift is left or right
            int modifier = std::uniform_int_distribution<int>(0, 1)(gen);

            int second_position = position + distance * (modifier ? -1 : 1);
            //will clamp the value between first and last vector positions
            second_position = std::min<int>(std::max<int>(0, second_position), child_code.occ_vector.size()-1);

            auto first_iterator  = child_code.occ_vector.begin() + position;
            auto second_iterator = child_code.occ_vector.begin() + second_position;

            int i = 0;
            while (i < group_size && first_iterator != child_code.occ_vector.end() && second_iterator != child_code.occ_vector.end()) {
                std::swap(*first_iterator, *second_iterator);

                i++;
                ++second_iterator;
                ++first_iterator;
            }
        }
            break;
    }
    return child_code;
}

OrganismConstructionCode::OrganismConstructionCode(const OrganismConstructionCode &parent_code) {
    occ_vector = std::vector(parent_code.occ_vector);
}

//OrganismConstructionCode::OrganismConstructionCode(OrganismConstructionCode &&code_to_move) noexcept {
//    occ_vector = std::move(code_to_move.occ_vector);
//}
