//
// Created by spaceeye on 15.09.22.
//

#include "OrganismConstructionCode.h"


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

        switch (instruction) {
            case OCCInstruction::ShiftUp:
            case OCCInstruction::ShiftUpLeft:
            case OCCInstruction::ShiftLeft:
            case OCCInstruction::ShiftLeftDown:
            case OCCInstruction::ShiftDown:
            case OCCInstruction::ShiftDownRight:
            case OCCInstruction::ShiftRight:
            case OCCInstruction::ShiftUpRight: {
                auto shift = shift_values[static_cast<int>(instruction)];

                if (!last_instruction) {
                    auto next_instruction = occ_vector[i + 1];

                    switch (next_instruction) {
                        case OCCInstruction::SetBlockMouth:
                        case OCCInstruction::SetBlockProducer:
                        case OCCInstruction::SetBlockMover:
                        case OCCInstruction::SetBlockKiller:
                        case OCCInstruction::SetBlockArmor:
                        case OCCInstruction::SetBlockEye:
                            if (x + shift[0] < edges[0]) { edges[0] = x + shift[0];}
                            if (x + shift[0] > edges[1]) { edges[1] = x + shift[0];}
                            if (y + shift[1] < edges[2]) { edges[2] = y + shift[1];}
                            if (y + shift[1] > edges[3]) { edges[3] = y + shift[1];}
                            continue;
                        default:
                            break;
                    }
                }
                x += shift[0];
                y += shift[1];

                if (x < edges[0]) { edges[0] = x;}
                if (x > edges[1]) { edges[1] = x;}
                if (y < edges[2]) { edges[2] = y;}
                if (y > edges[3]) { edges[3] = y;}
            }
                break;
            case OCCInstruction::ResetToOrigin:
                x = origin_x;
                y = origin_y;
                break;
            case OCCInstruction::SetOrigin:
                origin_x = x;
                origin_y = y;
                break;
            default: break;
        }
    }

    return edges;
}

SerializedOrganismStructureContainer *OrganismConstructionCode::compile_code(OCCLogicContainer & occ_c) {
    auto * container = new SerializedOrganismStructureContainer();

    auto edges = calculate_construction_edges();
    auto edge_width = std::abs(edges[0]) + std::abs(edges[1]) + 1;
    auto edge_height = std::abs(edges[2]) + std::abs(edges[3]) + 1;
    if (edge_width > occ_c.occ_width || edge_height > occ_c.occ_height) {
        occ_c.occ_width  = edge_width;
        occ_c.occ_height = edge_height;
        occ_c.occ_main_block_construction_space.resize(edge_width*edge_height);
        //TODO i don't want to bother with calculating additional values right now.
        occ_c.occ_producing_space.resize((edge_width + 2) * (edge_height + 2));
        occ_c.occ_eating_space   .resize((edge_width + 2) * (edge_height + 2));
        occ_c.occ_killing_space  .resize((edge_width + 2) * (edge_height + 2));
    }

    container->organism_blocks = compile_base_structure(container, occ_c, edges);
    edges[0]--;edges[1]++;edges[2]--;edges[3]++;

    auto spaces = compile_spaces(occ_c, edges, container->organism_blocks, container);
    container->producing_space = std::move(std::get<0>(spaces));
    container->eating_space    = std::move(std::get<1>(spaces));
    container->killing_space   = std::move(std::get<2>(spaces));

    occ_c.blocks.clear();
    return container;
}

void set_block(int x, int y, BlockTypes type, Rotation rotation, OCCLogicContainer &occ_c,
               std::vector<SerializedOrganismBlockContainer> &temp_blocks,
               SerializedOrganismStructureContainer *structure_container, int center_x, int center_y) {
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
        switch (block.type) {
            case BlockTypes::MouthBlock:    structure_container->mouth_blocks--    ; break;
            case BlockTypes::ProducerBlock: structure_container->producer_blocks-- ; break;
            case BlockTypes::MoverBlock:    structure_container->mover_blocks--    ; break;
            case BlockTypes::KillerBlock:   structure_container->killer_blocks--   ; break;
            case BlockTypes::ArmorBlock:    structure_container->armor_blocks--    ; break;
            case BlockTypes::EyeBlock:      structure_container->eye_blocks--      ; break;
            case BlockTypes::EmptyBlock:
            case BlockTypes::FoodBlock:
            case BlockTypes::WallBlock:
                break;
        }
        temp_blocks[block.parent_block_pos].type = type;
        temp_blocks[block.parent_block_pos].rotation = rotation;
        block.type = type;
    }

    switch (type) {
        case BlockTypes::MouthBlock:    structure_container->mouth_blocks++    ; break;
        case BlockTypes::ProducerBlock: structure_container->producer_blocks++ ; break;
        case BlockTypes::MoverBlock:    structure_container->mover_blocks++    ; break;
        case BlockTypes::KillerBlock:   structure_container->killer_blocks++   ; break;
        case BlockTypes::ArmorBlock:    structure_container->armor_blocks++    ; break;
        case BlockTypes::EyeBlock:      structure_container->eye_blocks++      ; break;
        case BlockTypes::EmptyBlock:
        case BlockTypes::FoodBlock:
        case BlockTypes::WallBlock:
            break;
    }
}

void set_rotation(int x, int y, Rotation rotation, OCCLogicContainer &occ_c,
                  std::vector<SerializedOrganismBlockContainer> &temp_blocks) {
    auto & block = occ_c.occ_main_block_construction_space[x + y * occ_c.occ_width];
    if (block.counter == occ_c.main_counter) {
        temp_blocks[block.parent_block_pos].rotation = rotation;
    }
}

std::vector<SerializedOrganismBlockContainer>
OrganismConstructionCode::compile_base_structure(SerializedOrganismStructureContainer *container,
                                                 OCCLogicContainer &occ_c, std::array<int, 4> edges) {
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
        auto next_instruction = OCCInstruction::ShiftUp;
        //if not last instruction
        //is needed for shift instruction logic.
        if (i != occ_vector.size()-1) {
            next_instruction = occ_vector[i+1];
        }

        switch (instruction) {
            case OCCInstruction::ShiftUp:
            case OCCInstruction::ShiftUpLeft:
            case OCCInstruction::ShiftLeft:
            case OCCInstruction::ShiftLeftDown:
            case OCCInstruction::ShiftDown:
            case OCCInstruction::ShiftDownRight:
            case OCCInstruction::ShiftRight:
            case OCCInstruction::ShiftUpRight: {
                //if next instruction is set block, then do not shift the cursor,
                // but place block on shifted position from cursor and increment i so that it will not get set block instruction
                bool pass = false;
                shift_instruction_part(container, occ_c, shift_values, shift, base_rotation, cursor_x, cursor_y,
                                       center_x, center_y,
                                       instruction, next_instruction, blocks, i, pass);
                if (pass) { continue; }
                shift = shift_values[static_cast<int>(instruction)];
                cursor_x += shift[0];
                cursor_y += shift[1];
            }
                break;

            case OCCInstruction::ApplyRotationUp:    base_rotation = Rotation::UP;    break;
            case OCCInstruction::ApplyRotationLeft:  base_rotation = Rotation::LEFT;  break;
            case OCCInstruction::ApplyRotationDown:  base_rotation = Rotation::DOWN;  break;
            case OCCInstruction::ApplyRotationRight: base_rotation = Rotation::RIGHT; break;

            case OCCInstruction::SetRotationUp:    set_rotation(cursor_x, cursor_y, Rotation::UP,    occ_c, blocks);break;
            case OCCInstruction::SetRotationLeft:  set_rotation(cursor_x, cursor_y, Rotation::LEFT,  occ_c, blocks);break;
            case OCCInstruction::SetRotationDown:  set_rotation(cursor_x, cursor_y, Rotation::DOWN,  occ_c, blocks);break;
            case OCCInstruction::SetRotationRight: set_rotation(cursor_x, cursor_y, Rotation::RIGHT, occ_c, blocks);break;

            case OCCInstruction::ResetToOrigin:
                cursor_x = origin_x;
                cursor_y = origin_y;
                break;
            case OCCInstruction::SetOrigin:
                origin_x = cursor_x;
                origin_y = cursor_y;
                break;

            case OCCInstruction::SetBlockMouth:    set_block(cursor_x, cursor_y, BlockTypes::MouthBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetBlockProducer: set_block(cursor_x, cursor_y, BlockTypes::ProducerBlock, base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetBlockMover:    set_block(cursor_x, cursor_y, BlockTypes::MoverBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetBlockKiller:   set_block(cursor_x, cursor_y, BlockTypes::KillerBlock,   base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetBlockArmor:    set_block(cursor_x, cursor_y, BlockTypes::ArmorBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetBlockEye:      set_block(cursor_x, cursor_y, BlockTypes::EyeBlock,      base_rotation, occ_c, blocks, container, center_x, center_y);break;

            case OCCInstruction::SetUnderBlockMouth:    set_block(cursor_x, cursor_y, BlockTypes::MouthBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetUnderBlockProducer: set_block(cursor_x, cursor_y, BlockTypes::ProducerBlock, base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetUnderBlockMover:    set_block(cursor_x, cursor_y, BlockTypes::MoverBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetUnderBlockKiller:   set_block(cursor_x, cursor_y, BlockTypes::KillerBlock,   base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetUnderBlockArmor:    set_block(cursor_x, cursor_y, BlockTypes::ArmorBlock,    base_rotation, occ_c, blocks, container, center_x, center_y);break;
            case OCCInstruction::SetUnderBlockEye:      set_block(cursor_x, cursor_y, BlockTypes::EyeBlock,      base_rotation, occ_c, blocks, container, center_x, center_y);break;
        }
    }

    return blocks;
}

void OrganismConstructionCode::shift_instruction_part(SerializedOrganismStructureContainer *container,
                                                      OCCLogicContainer &occ_c,
                                                      const std::array<std::array<int, 2>, 8> &shift_values,
                                                      std::array<int, 2> &shift, Rotation &base_rotation, int cursor_x,
                                                      int cursor_y, int center_x, int center_y,
                                                      const OCCInstruction &instruction,
                                                      const OCCInstruction &next_instruction,
                                                      std::vector<SerializedOrganismBlockContainer> &blocks, int &i,
                                                      bool &pass) {
    switch (next_instruction) {
        case OCCInstruction::SetBlockMouth:
        case OCCInstruction::SetBlockProducer:
        case OCCInstruction::SetBlockMover:
        case OCCInstruction::SetBlockKiller:
        case OCCInstruction::SetBlockArmor:
        case OCCInstruction::SetBlockEye:
            shift = shift_values[static_cast<int>(instruction)];
            set_block(cursor_x + shift[0], cursor_y + shift[1],
                      //set block instruction are in the same order as block types.
                      static_cast<BlockTypes>(static_cast<int>(next_instruction) - 18 + 1),
                      base_rotation, occ_c, blocks, container, center_x, center_y);
            i++;
            pass = true;
            break;
        default:
            break;
    }
}

//Will compile spaces all at the same time in one go.
std::tuple<std::vector<std::vector<SerializedAdjacentSpaceContainer>>, std::vector<SerializedAdjacentSpaceContainer>, std::vector<SerializedAdjacentSpaceContainer>>
OrganismConstructionCode::compile_spaces(OCCLogicContainer &occ_c, std::array<int, 4> edges,
                                         std::vector<SerializedOrganismBlockContainer> &organism_blocks,
                                         SerializedOrganismStructureContainer *container) {
    std::tuple<std::vector<std::vector<SerializedAdjacentSpaceContainer>>, std::vector<SerializedAdjacentSpaceContainer>, std::vector<SerializedAdjacentSpaceContainer>> spaces;

    auto shifting_positions = std::array<std::array<int, 2>, 4> {
        std::array<int, 2>{ 0,-1},
        std::array<int, 2>{-1, 0},
        std::array<int, 2>{ 0, 1},
        std::array<int, 2>{ 1, 0},
    };

    occ_c.spaces_counter++;

    auto &producing_space = std::get<0>(spaces);
    auto &eating_space    = std::get<1>(spaces);
    auto &killer_space    = std::get<2>(spaces);

    auto temp_producing_space = std::vector<OCCSerializedProducingSpace>();

    producing_space.resize(container->producer_blocks);
    //TODO probably not very space efficient
    eating_space.reserve(container->mouth_blocks*4);
    killer_space.reserve(container->killer_blocks*4);

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
                    // if eating space is not already occupied and there is no block existing on this pos in main_space.
                    if (x_-1 <= 0 || y_-1 <= 0 || x_-1 >= occ_c.occ_width-1 || y_-1 >= occ_c.occ_height-1 ||
                        occ_c.occ_producing_space[x_ + y_ * (occ_c.occ_height + 2)].counter != occ_c.spaces_counter
                        && occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_height].counter != occ_c.main_counter) {
                        temp_producing_space.emplace_back(producer, x_-center_x, y_-center_y);
                        occ_c.occ_producing_space[x_ + y_ * (occ_c.occ_height + 2)].parent_block_pos = eating_space.size() - 1;
                    } else if (occ_c.occ_producing_space[x_ + y_ * (occ_c.occ_height + 2)].counter == occ_c.spaces_counter) {
                        temp_producing_space[occ_c.occ_producing_space[x_ + y_ * (occ_c.occ_height + 2)].parent_block_pos].producer = producer;
                    }
                }
                break;
            case BlockTypes::MouthBlock:
                for (auto & shift: shifting_positions) {
                    int x_ = x + shift[0];
                    int y_ = y + shift[1];
                    // if eating space is not already occupied and there is no block existing on this pos in main_space.
                    if (x_-1 < 0 || y_-1 < 0 || x_-1 >= occ_c.occ_width || y_-1 >= occ_c.occ_height ||
                        occ_c.occ_eating_space[x_ + y_ * (occ_c.occ_height + 2)].counter != occ_c.spaces_counter
                        && occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_height].counter != occ_c.main_counter) {
                        eating_space.emplace_back(x_-center_x, y_-center_y);
                        occ_c.occ_eating_space[x_ + y_ * (occ_c.occ_height + 2)].parent_block_pos = eating_space.size() - 1;
                    }
                }
                break;
            case BlockTypes::KillerBlock:
                for (auto & shift: shifting_positions) {
                    int x_ = x + shift[0];
                    int y_ = y + shift[1];
                    // if eating space is not already occupied and there is no block existing on this pos in main_space.
                    if (x_-1 < 0 || y_-1 < 0 || x_-1 >= occ_c.occ_width || y_-1 >= occ_c.occ_height ||
                        occ_c.occ_killing_space[x_ + y_ * (occ_c.occ_height + 2)].counter != occ_c.spaces_counter
                        && occ_c.occ_main_block_construction_space[(x_-1) + (y_-1) * occ_c.occ_height].counter != occ_c.main_counter) {
                        killer_space.emplace_back(x_-center_x, y_-center_y);
                        occ_c.occ_killing_space[x_ + y_ * (occ_c.occ_height + 2)].parent_block_pos = killer_space.size() - 1;
                    }
                }
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

    return spaces;
}

std::vector<OCCInstruction> create_random_group(int group_size, OCCParameters &occp, lehmer64 &gen) {
    std::vector<OCCInstruction> group;
    group.reserve(group_size);

    for (int i = 0; i < group_size; i++) {
        if (occp.uniform_occ_instructions_mutation) {
            group.emplace_back(static_cast<OCCInstruction>(std::uniform_int_distribution<int>(0, 23)(gen)));
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
        //insert group into sequence
        case OCCMutations::InsertRandom: {
            auto group = create_random_group(group_size, occp, gen);
            int position = std::uniform_int_distribution<int>(0, child_code.occ_vector.size()-1)(gen);

            child_code.occ_vector.insert(child_code.occ_vector.begin()+position, group.begin(), group.end());
        }
            break;
        //Overwrite existing part of a sequence with a group
        case OCCMutations::ChangeRandom: {
            if (child_code.occ_vector.size() == 1) { break;}
            auto group = create_random_group(group_size, occp, gen);
            int starting_position = 1;
            switch (group[0]) {
                case OCCInstruction::SetBlockMouth:
                case OCCInstruction::SetBlockProducer:
                case OCCInstruction::SetBlockMover:
                case OCCInstruction::SetBlockKiller:
                case OCCInstruction::SetBlockArmor:
                case OCCInstruction::SetBlockEye:
                    starting_position = 0;
                    break;
                default: break;
            }

            int position = std::uniform_int_distribution<int>(starting_position, child_code.occ_vector.size()-1)(gen);

            int iterated = 0;
            for (auto iterator = child_code.occ_vector.begin()+position; iterator != child_code.occ_vector.end() && iterated < group.size(); iterator++) {
                iterated++;
                (*iterator) = group[iterated-1];
            }
        }
            break;
        //delete part of a sequence with group_size
        case OCCMutations::DeleteRandom: {
            if (child_code.occ_vector.size() == 1) { break;}
            int position = std::uniform_int_distribution<int>(1, child_code.occ_vector.size()-1)(gen);
            int allowed_erasing = std::min<int>(group_size, child_code.occ_vector.size()-position-1);

            auto position_iterator = child_code.occ_vector.begin()+position;
            child_code.occ_vector.erase(position_iterator, position_iterator+allowed_erasing);
        }
            break;
        //shift instructions with size of group_size to the left or right by some distance
        case OCCMutations::MoveRandom: {
            if (child_code.occ_vector.size() == 1) { break;}
            int position = std::uniform_int_distribution<int>(1, child_code.occ_vector.size()-1)(gen);
            int distance;

            if (occp.uniform_move_distance) {
                distance = std::uniform_int_distribution<int>(1, occp.max_distance)(gen);
            } else {
                distance = occp.move_distance_mutation_discrete_distribution(gen);
            }

            //will determine whenever the shift is left or right
            int modifier = std::uniform_int_distribution<int>(0, 1)(gen);

            int second_position = position + distance * (modifier ? -1 : 1);
            //will clamp the value between first and last vector positions
            second_position = std::min<int>(std::max<int>(1, second_position), occ_vector.size()-1);

            auto first_iterator  = occ_vector.begin() + position;
            auto second_iterator = occ_vector.begin() + second_position;

            int i = 0;
            while (i < group_size && first_iterator != occ_vector.end() && second_iterator != occ_vector.end()) {
                std::swap(*first_iterator, *second_iterator);

                i++;
                second_iterator++;
                first_iterator++;
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
