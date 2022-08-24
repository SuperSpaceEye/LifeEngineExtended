// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.05.2022.
//

#include "SimulationEngineSingleThread.h"
#include "../OrganismsController.h"

void SimulationEngineSingleThread::single_threaded_tick(EngineDataContainer * dc, SimulationParameters * sp, lehmer64 *gen) {
    if (sp->eat_then_produce) {
        for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {eat_food(dc, sp, &organism);}}
        for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {produce_food(dc, sp, &organism, *gen);}}
    } else {
        for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {produce_food(dc, sp, &organism, *gen);}}
        for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {eat_food(dc, sp, &organism);}}
    }

    for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {apply_damage(dc, sp, &organism);}}

    for (int i = dc->stc.last_alive_position; i >= 0; i--) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {tick_lifetime(dc, &organism);}}

    dc->stc.organisms_observations.clear();

    reserve_observations(dc->stc.organisms_observations, dc->stc.organisms, dc);
    for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) { get_observations(dc, sp, &organism, dc->stc.organisms_observations);}}

    for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {organism.think_decision(dc->stc.organisms_observations[i], gen);}}
    for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {make_decision(dc, sp, &organism, gen);}}

    OrganismsController::precise_sort_dead_organisms(*dc);
    for (int i = 0; i <= dc->stc.last_alive_position; i++) {auto & organism = dc->stc.organisms[i]; if (!organism.is_dead) {try_make_child(dc, sp, &organism, gen);}}
}

void SimulationEngineSingleThread::place_organism(EngineDataContainer *dc, Organism *organism) {
    for (auto &block: organism->anatomy._organism_blocks) {
        place_block_on_grid(dc, organism, block);
    }
}

void SimulationEngineSingleThread::produce_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, lehmer64 &gen) {
    if (organism->anatomy._producer_blocks == 0) {return;}
    if (organism->anatomy._mover_blocks > 0 && !sp->movers_can_produce_food) {return;}
    //TODO delete?
    if (organism->lifetime % sp->produce_food_every_n_life_ticks != 0) {return;}

    if (sp->simplified_food_production) {
        produce_food_simplified(dc, sp, organism, gen, organism->multiplier);
    } else {
        produce_food_complex(dc, sp, organism, gen, organism->multiplier);
    }
}

void SimulationEngineSingleThread::produce_food_simplified(EngineDataContainer *dc, SimulationParameters *sp,
                                                           Organism *organism, lehmer64 &gen, float multiplier) {
    for (auto & pr: organism->anatomy._producing_space) {
        for (auto &pc: pr) {
            auto * w_block = &dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y];
            if (w_block->type != BlockTypes::EmptyBlock) {continue;}
            if (std::uniform_real_distribution<float>(0, 1)(gen) < sp->food_production_probability * multiplier) {
                w_block->type = BlockTypes::FoodBlock;
                if (sp->stop_when_one_food_generated) { return;}
                continue;
            }
        }
    }
}

void SimulationEngineSingleThread::produce_food_complex(EngineDataContainer *dc, SimulationParameters *sp,
                                                        Organism *organism, lehmer64 &gen, float multiplier) {
    for (auto & pr: organism->anatomy._producing_space) {
        //First checks if producer can produce
        if (std::uniform_real_distribution<float>(0, 1)(gen) > sp->food_production_probability * multiplier) { continue;}

        //Then selects one space to produce food.
        auto & pc = pr[std::uniform_int_distribution<int>(0, pr.size()-1)(gen)];

        auto * w_block = &dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y];

        //if space is occupied, then do nothing
        if (w_block->type != BlockTypes::EmptyBlock) { continue;}
        w_block->type = BlockTypes::FoodBlock;
        if (sp->stop_when_one_food_generated) { return;}
    }
}

void SimulationEngineSingleThread::eat_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto & pc: organism->anatomy._eating_space) {
        auto * w_block = &dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y];
        if (w_block->type == BlockTypes::FoodBlock) {
            w_block->type = BlockTypes::EmptyBlock;
            organism->food_collected++;
        }
    }
}

void SimulationEngineSingleThread::tick_lifetime(EngineDataContainer *dc, Organism *organism) {
    organism->lifetime++;
    if (organism->lifetime > organism->max_lifetime || organism->damage > organism->life_points) {
        organism->kill_organism(*dc);
        for (auto & block: organism->anatomy._organism_blocks) {
            auto * w_block = &dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y];
            w_block->type = BlockTypes::FoodBlock;
            w_block->organism_index = -1;
        }
    }
}

void SimulationEngineSingleThread::apply_damage(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto &block: organism->anatomy._killing_space) {
        auto world_block = dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][
                organism->y + block.get_pos(organism->rotation).y];
        switch (world_block.type) {
            case BlockTypes::EmptyBlock:
            case BlockTypes::FoodBlock:
            case BlockTypes::WallBlock:
            case BlockTypes::ArmorBlock:
                continue;
            default:
                break;
        }
        auto * world_organism = OrganismsController::get_organism_by_index(world_block.organism_index, *dc);
        if (world_organism == nullptr) { continue;}
        if (sp->on_touch_kill) { world_organism->damage = world_organism->life_points + 1; break; }
        world_organism->damage += sp->killer_damage_amount;
    }
}

void SimulationEngineSingleThread::reserve_observations(std::vector<std::vector<Observation>> &observations,
                                                        std::vector<Organism> &organisms,
                                                        EngineDataContainer *dc) {
    observations.reserve(dc->stc.last_alive_position+1);
    for (int i = 0; i <= dc->stc.last_alive_position; i++) {
        auto & organism = dc->stc.organisms[i];
        if (organism.is_dead) { observations.emplace_back(); continue;}

        //if organism has no eyes, movers or is moving, then do not observe.
        if (organism.anatomy._eye_blocks > 0 && organism.anatomy._mover_blocks > 0 && organism.move_counter == 0) {
            observations.emplace_back(std::vector<Observation>(organism.anatomy._eye_blocks));
        } else {
            observations.emplace_back();
        }
    }
}

void SimulationEngineSingleThread::get_observations(EngineDataContainer *dc, SimulationParameters *sp,
                                                    Organism *organism,
                                                    std::vector<std::vector<Observation>> &organism_observations) {
    if (organism->anatomy._eye_blocks <= 0 || organism->anatomy._mover_blocks <= 0) {return;}
    if (organism->move_counter != 0) {return;}
    auto eye_i = -1;
    //TODO inefficient
    for (auto & block: organism->anatomy._organism_blocks) {
        if (block.type != BlockTypes::EyeBlock) {continue;}
        eye_i++;
        auto pos_x = organism->x + block.get_pos(organism->rotation).x;
        auto pos_y = organism->y + block.get_pos(organism->rotation).y;
        // getting global rotation on a simulation grid
        auto block_rotation = block.get_block_rotation_on_grid(organism->rotation);

        auto offset_x = 0;
        auto offset_y = 0;

        switch (block_rotation) {
            case Rotation::UP:
                offset_y = -1;
                break;
            case Rotation::LEFT:
                offset_x = -1;
                break;
            case Rotation::DOWN:
                offset_y = 1;
                break;
            case Rotation::RIGHT:
                offset_x = 1;
                break;
        }

        auto last_observation = Observation{BlockTypes::EmptyBlock, 0, block.rotation};

        for (int i = 1; i < sp->look_range; i++) {
            pos_x += offset_x;
            pos_y += offset_y;

            last_observation.type = dc->CPU_simulation_grid[pos_x][pos_y].type;
            last_observation.distance = i;

            //TODO maybe switch?
            if (last_observation.type == BlockTypes::WallBlock) {break;}
            if (last_observation.type == BlockTypes::FoodBlock) {break;}
            if (last_observation.type != BlockTypes::EmptyBlock) {
                if (!sp->organism_self_blocks_block_sight && dc->CPU_simulation_grid[pos_x][pos_y].organism_index == organism->vector_index) {
                    continue;
                }
                break;
            }
        }
        organism_observations[organism->vector_index][eye_i] = last_observation;
    }
}

void SimulationEngineSingleThread::rotate_organism(EngineDataContainer *dc, Organism *organism, BrainDecision decision,
                                                   SimulationParameters *sp) {
    auto new_int_rotation = static_cast<uint_fast8_t>(organism->rotation);
    switch (decision) {
        case BrainDecision::RotateLeft:
            new_int_rotation += 1;
            break;
        case BrainDecision::RotateRight:
            new_int_rotation -= 1;
            break;
        case BrainDecision::Flip:
            new_int_rotation += 2;
            break;
        default: break;
    }

    auto new_rotation = static_cast<Rotation>(new_int_rotation%4);

    //checks if space for organism is empty, or contains itself
    for (auto & block: organism->anatomy._organism_blocks) {
        auto * w_block = &dc->CPU_simulation_grid[organism->x + block.get_pos(new_rotation).x][organism->y + block.get_pos(new_rotation).y];

        if (check_if_block_out_of_bounds(dc, organism, block, new_rotation)) { return;}

        if (sp->food_blocks_movement) {
            if (w_block->type != BlockTypes::EmptyBlock && w_block->organism_index != organism->vector_index) {
                return;
            }
        } else {
            if ((w_block->type != BlockTypes::EmptyBlock && w_block->type != BlockTypes::FoodBlock) && w_block->organism_index != organism->vector_index) {
                return;
            }
        }
    }

    for (auto & block: organism->anatomy._organism_blocks) {
        auto pos = block.get_pos(organism->rotation);
        auto * w_block = &dc->CPU_simulation_grid[organism->x + pos.x][organism->y + pos.y];
        w_block->type = BlockTypes::EmptyBlock;
        w_block->organism_index = -1;
    }

    //If there is a place for rotated organism, then rotation can happen
    organism->rotation = new_rotation;
    for (auto & block: organism->anatomy._organism_blocks) {
        place_block_on_grid(dc, organism, block);

        auto pos = block.get_pos(organism->rotation);
        auto * w_block = &dc->CPU_simulation_grid[organism->x + pos.x][organism->y + pos.y];
        w_block->type = block.type;
        if (organism->rotation == Rotation::LEFT || organism->rotation == Rotation::RIGHT) {
            w_block->rotation = get_global_rotation((Rotation)(((int)block.rotation + 2)%4), organism->rotation);
        } else {
            w_block->rotation = get_global_rotation(block.rotation, organism->rotation);
        }
        w_block->organism_index = organism->vector_index;
    }
}

void SimulationEngineSingleThread::place_block_on_grid(EngineDataContainer *dc, Organism *organism,
                                                       SerializedOrganismBlockContainer &block) {
    auto pos = block.get_pos(organism->rotation);
    auto * w_block = &dc->CPU_simulation_grid[organism->x + pos.x][organism->y + pos.y];
    w_block->type = block.type;
    //TODO I have no idea why, but if rotation of organism is left or right, the rotation on the grid is wrong.
    if (organism->rotation == Rotation::LEFT || organism->rotation == Rotation::RIGHT) {
        w_block->rotation = get_global_rotation((Rotation)(((int)block.rotation + 2)%4), organism->rotation);
    } else {
        w_block->rotation = get_global_rotation(block.rotation, organism->rotation);
    }
    w_block->organism_index = organism->vector_index;
}

void SimulationEngineSingleThread::move_organism(EngineDataContainer *dc, Organism *organism, BrainDecision decision,
                                                 SimulationParameters *sp) {
    // rotates movement relative to simulation grid

    int new_x = organism->x;
    int new_y = organism->y;

    switch (decision) {
        case BrainDecision::MoveUp:    new_y -= 1; break;
        case BrainDecision::MoveLeft:  new_x -= 1; break;
        case BrainDecision::MoveDown:  new_y += 1; break;
        case BrainDecision::MoveRight: new_x += 1; break;
        default: break;
    }

    //Organism can move only by 1 block a simulation tick, so it will be stopped by a wall and doesn't need an out-of-bounds check.
    for (auto & block: organism->anatomy._organism_blocks) {
        auto pos = block.get_pos(organism->rotation);
        auto * w_block = &dc->CPU_simulation_grid[new_x + pos.x][new_y + pos.y];
        if (sp->food_blocks_movement) {
            if (w_block->type != BlockTypes::EmptyBlock &&
                w_block->organism_index != organism->vector_index) {
                return;
            }
        } else {
            if ((w_block->type != BlockTypes::EmptyBlock && w_block->type != BlockTypes::FoodBlock) && w_block->organism_index != organism->vector_index) {
                return;
            }
        }
    }

    for (auto & block: organism->anatomy._organism_blocks) {
        auto pos = block.get_pos(organism->rotation);
        auto * w_block = &dc->CPU_simulation_grid[organism->x + pos.x][organism->y + pos.y];
        w_block->type = BlockTypes::EmptyBlock;
        w_block->organism_index = -1;
    }

    for (auto & block: organism->anatomy._organism_blocks) {
        auto pos = block.get_pos(organism->rotation);
        auto * w_block = &dc->CPU_simulation_grid[new_x + pos.x][new_y + pos.y];
        w_block->type = block.type;
        w_block->rotation = get_global_rotation(block.rotation,organism->rotation);
//        if (block.type == BlockTypes::EyeBlock) {
//            w_block->rotation = get_global_rotation(block.rotation,organism->rotation);
//        }
        w_block->organism_index = organism->vector_index;
    }

    organism->x = new_x;
    organism->y = new_y;
}

void SimulationEngineSingleThread::make_decision(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, lehmer64 *gen) {
    switch (organism->last_decision.decision) {
        case BrainDecision::MoveUp:
        case BrainDecision::MoveDown:
        case BrainDecision::MoveLeft:
        case BrainDecision::MoveRight:
            if (organism->anatomy._mover_blocks > 0) {
                move_organism(dc, organism, organism->last_decision.decision, sp);
                if (sp->rotate_every_move_tick && organism->anatomy._mover_blocks > 0 && sp->runtime_rotation_enabled) {
                    rotate_organism(dc, organism,
                                    static_cast<BrainDecision>(std::uniform_int_distribution<int>(4, 6)(*gen)),
                                    sp);
                }
                organism->move_counter++;
            }
            break;
        default: break;
    }
    if ((organism->move_counter >= organism->move_range) || (sp->set_fixed_move_range && sp->min_move_range == organism->move_counter)) {
        organism->move_counter = 0;
        if (!sp->rotate_every_move_tick && organism->anatomy._mover_blocks > 0 && sp->runtime_rotation_enabled) {
            rotate_organism(dc, organism, static_cast<BrainDecision>(std::uniform_int_distribution<int>(4, 6)(*gen)),
                            sp);
        }
    }
}

void SimulationEngineSingleThread::try_make_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                  lehmer64 *gen) {
    if (organism->child_pattern_index < 0) { organism->child_pattern_index = organism->create_child(gen, *dc);}
    // if max_organisms < 0, then unlimited.
    if (dc->max_organisms >= 0 && dc->stc.num_alive_organisms >= dc->max_organisms) {return;}
    auto child_food_needed = OrganismsController::get_child_organism_by_index(organism->child_pattern_index, *dc)->food_needed;
    if (organism->food_collected < child_food_needed) {return;}
    if (sp->failed_reproduction_eats_food) {organism->food_collected -= child_food_needed;}
    place_child(dc, sp, organism, gen);
}

void SimulationEngineSingleThread::place_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                               lehmer64 *gen) {
    auto to_place = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*gen));
    Rotation rotation;
    if (sp->reproduction_rotation_enabled) {
        rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*gen));
    } else {
        rotation = Rotation::UP;
    }

    int distance = sp->min_reproducing_distance;
    if (!sp->reproduction_distance_fixed) {
        distance = std::uniform_int_distribution<int>(sp->min_reproducing_distance, sp->max_reproducing_distance)(*gen);
    }

    auto * child_pattern = OrganismsController::get_child_organism_by_index(organism->child_pattern_index, *dc);

    child_pattern->rotation = rotation;

    // Old and new versions drastically influence evolution, so I will add both.
    if (sp->use_new_child_pos_calculator) {
        new_child_pos_calculator(organism, to_place, distance, *dc);
    } else {
        old_child_pos_calculator(organism, to_place, distance, *dc);
    }

    if (sp->check_if_path_is_clear) {
        int c_distance = std::abs(organism->x - child_pattern->x) + std::abs(organism->y - child_pattern->y);

        for (auto & block: child_pattern->anatomy._organism_blocks) {
            if (!path_is_clear(organism->x + block.get_pos(rotation).x,
                               organism->y + block.get_pos(rotation).y,
                               to_place,
                               c_distance,
                               organism->vector_index, dc, sp)) {
                return;}
        }
    }

    //checking, if there is space for a child
    for (auto & block: child_pattern->anatomy._organism_blocks) {
        if (check_if_block_out_of_bounds(dc, child_pattern, block, child_pattern->rotation)) {return;}

        auto * w_block = &dc->CPU_simulation_grid[child_pattern->x + block.get_pos(child_pattern->rotation).x]
                                    [child_pattern->y + block.get_pos(child_pattern->rotation).y];

        if (sp->food_blocks_reproduction && w_block->type != BlockTypes::EmptyBlock
            ||
            !sp->food_blocks_reproduction && !(w_block->type == BlockTypes::EmptyBlock ||
            w_block->type == BlockTypes::FoodBlock)
            )
        {return;}
    }

    //only deduct food when reproduction is successful and flag is false
    if (!sp->failed_reproduction_eats_food) {
        organism->food_collected -= child_pattern->food_needed;
    }

    //It is needed because pointer to organism can become invalid during emplace_child_organism_to_main_vector if organism vector resizes.
    auto main_organism_index = organism->vector_index;
    auto child_index = OrganismsController::emplace_child_organisms_to_main_vector(child_pattern, *dc);
    place_organism(dc, OrganismsController::get_organism_by_index(child_index, *dc));
    OrganismsController::get_organism_by_index(main_organism_index, *dc)->child_pattern_index = -1;
}

void SimulationEngineSingleThread::new_child_pos_calculator(Organism *organism, const Rotation to_place, int distance,
                                                            EngineDataContainer &edc) {
    auto o_min_y = 0;
    auto o_min_x = 0;
    auto o_max_y = 0;
    auto o_max_x = 0;

    auto c_min_y = 0;
    auto c_min_x = 0;
    auto c_max_y = 0;
    auto c_max_x = 0;

    auto * child_pattern = OrganismsController::get_child_organism_by_index(organism->child_pattern_index, edc);

    //To get the position of a child on an axis you need to know the largest/smallest position of organism blocks on an axis
    // and smallest/largest position of a child organism block + -1/+1 to compensate for organism center and + -/+ distance.
    //For example, if a child is reproduced up of organism, you need the upmost coordinate among organism blocks and
    // lowest coordinate of child organism blocks with -1 to compensate for organism center. With that, child will spawn
    // basically attached to the parent organism, so then you subtract the random distance.

    switch (to_place) {
        case Rotation::UP:
            o_min_y = INT32_MAX;
            c_max_y = INT32_MIN;
            for (auto & block: organism->anatomy._organism_blocks) {
                auto pos = block.get_pos(organism->rotation);
                if (pos.y < o_min_y) { o_min_y = pos.y;}
            }
            for (auto & block: child_pattern->anatomy._organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.y > c_max_y) { c_max_y = pos.y;}
            }
            o_min_y -= distance;
            c_max_y = -abs(c_max_y) - 1;
            break;
        case Rotation::LEFT:
            o_min_x = INT32_MAX;
            c_max_x = INT32_MIN;
            for (auto & block: organism->anatomy._organism_blocks) {
                auto pos = block.get_pos(organism->rotation);
                if (pos.x < o_min_x) { o_min_x = pos.x;}
            }
            for (auto & block: child_pattern->anatomy._organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.x > c_max_x) { c_max_x = pos.x;}
            }
            o_min_x -= distance;
            c_max_x = -abs(c_max_x) - 1;
            break;
        case Rotation::DOWN:
            o_max_y = INT32_MIN;
            c_min_y = INT32_MAX;
            for (auto & block: organism->anatomy._organism_blocks) {
                auto pos = block.get_pos(organism->rotation);
                if (pos.y > o_max_y) { o_max_y = pos.y;}
            }
            for (auto & block: child_pattern->anatomy._organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.y < c_min_y) { c_min_y = pos.y;}
            }
            o_max_y += distance;
            c_min_y = abs(c_min_y) + 1;
            break;
        case Rotation::RIGHT:
            o_max_x = INT32_MIN;
            c_min_x = INT32_MAX;
            for (auto & block: organism->anatomy._organism_blocks){
                auto pos = block.get_pos(organism->rotation);
                if (pos.x > o_max_x) { o_max_x = pos.x;}
            }
            for (auto & block: child_pattern->anatomy._organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.x < c_min_x) { c_min_x = pos.x;}
            }
            o_max_x += distance;
            c_min_x = abs(c_min_x) + 1;
            break;
    }

    child_pattern->x = organism->x + o_min_x + o_max_x + c_min_x + c_max_x;
    child_pattern->y = organism->y + o_min_y + o_max_y + c_min_y + c_max_y;
}

void SimulationEngineSingleThread::old_child_pos_calculator(Organism *organism, Rotation to_place, int distance,
                                                            EngineDataContainer &edc) {
    auto min_y = INT32_MAX;
    auto min_x = INT32_MAX;
    auto max_y = INT32_MIN;
    auto max_x = INT32_MIN;

    auto * child_pattern = OrganismsController::get_child_organism_by_index(organism->child_pattern_index, edc);

    switch (to_place) {
        case Rotation::UP:
            min_y = INT32_MAX;
            for (auto & block: organism->anatomy._organism_blocks) {
                if (block.get_pos(organism->rotation).y < min_y) {min_y = block.get_pos(organism->rotation).y;}
            }
            min_y -= distance;
            break;
        case Rotation::LEFT:
            min_x = INT32_MAX;
            for (auto & block: organism->anatomy._organism_blocks) {
                if (block.get_pos(organism->rotation).x < min_x) {min_x = block.get_pos(organism->rotation).x;}
            }
            min_x -= distance;
            break;
        case Rotation::DOWN:
            max_y = INT32_MIN;
            for (auto & block: organism->anatomy._organism_blocks) {
                if (block.get_pos(organism->rotation).y > max_y) {max_y = block.get_pos(organism->rotation).y;}
            }
            max_y += distance;
            break;
        case Rotation::RIGHT:
            max_x = INT32_MIN;
            for (auto & block: organism->anatomy._organism_blocks){
                if (block.get_pos(organism->rotation).x > max_x) {max_x = block.get_pos(organism->rotation).x;}
            }
            max_x += distance;
            break;
    }

    child_pattern->x = organism->x + min_x + max_x;
    child_pattern->y = organism->y + min_y + max_y;
}

bool SimulationEngineSingleThread::check_if_out_of_bounds(EngineDataContainer *dc, int x, int y) {
    return (x < 0 ||
            x > dc->simulation_width -1 ||
            y < 0 ||
            y > dc->simulation_height-1);
}

// if any is true, then check fails, else check succeeds
bool SimulationEngineSingleThread::check_if_block_out_of_bounds(EngineDataContainer *dc, Organism *organism,
                                                                BaseSerializedContainer &block, Rotation rotation) {
    return check_if_out_of_bounds(dc,
                                  organism->x + block.get_pos(rotation).x,
                                  organism->y + block.get_pos(rotation).y);
}

bool SimulationEngineSingleThread::path_is_clear(int x, int y, Rotation direction, int steps, int32_t allow_organism, EngineDataContainer *dc,
                   SimulationParameters *sp) {
    //"steps-1" because the final position will be checked either way
    for (int i = 0; i < steps-1; i++) {
        switch (direction) {
            case Rotation::UP:
                y -= 1;
                break;
            case Rotation::LEFT:
                x -= 1;
                break;
            case Rotation::DOWN:
                y += 1;
                break;
            case Rotation::RIGHT:
                x += 1;
                break;
        }
        if (check_if_out_of_bounds(dc, x, y)) {
            return false;
        };

        auto * block = &dc->CPU_simulation_grid[x][y];
        switch (block->type) {
            case BlockTypes::EmptyBlock:
                continue;
            case BlockTypes::WallBlock:
                return false;
            case BlockTypes::FoodBlock:
                //Big producers will not evolve if uncommented.
//                if (sp->food_blocks_reproduction) {
//                    return false;
//                }
                continue;
            default:
                if (block->organism_index == allow_organism) { continue;}
                return false;
        }
    }
    return true;
}

std::array<int, 4> SimulationEngineSingleThread::get_organism_dimensions(Organism * organism) {
    std::array<int, 4> output{INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN};
    auto & anatomy = organism->anatomy;

    for (auto & block: anatomy._organism_blocks) {
        if (block.relative_x < output[0]) {output[0] = block.relative_x;}
        if (block.relative_y < output[1]) {output[1] = block.relative_y;}
        if (block.relative_x > output[2]) {output[2] = block.relative_x;}
        if (block.relative_y > output[3]) {output[3] = block.relative_y;}
    }

    return output;
}
