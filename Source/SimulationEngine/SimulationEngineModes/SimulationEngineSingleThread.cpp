// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 16.05.2022.
//

#include "SimulationEngineSingleThread.h"
#include "../OrganismsController.h"

void
SimulationEngineSingleThread::single_threaded_tick(EngineDataContainer &edc, SimulationParameters &sp, lehmer64 &gen) {
    if (sp.eat_then_produce) {
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {eat_food(edc, sp, organism);}}
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {produce_food(edc, sp, organism, gen);}}
    } else {
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {produce_food(edc, sp, organism, gen);}}
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {eat_food(edc, sp, organism);}}
    }

    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {apply_damage(edc, sp, organism);}}

    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {
            tick_lifetime(
                    edc, organism, i, sp);}}

    edc.stc.organisms_observations.clear();

    reserve_observations(edc.stc.organisms_observations, edc.stc.organisms, edc);
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {get_observations(edc, sp, organism, edc.stc.organisms_observations);}}

    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {organism.think_decision(edc.stc.organisms_observations[i], gen);}}
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {make_decision(edc, sp, organism, gen);}}

    OrganismsController::try_compress_organisms(edc);
    //TODO the result of sorting doesn't need to be perfect, just good enough.
    //https://en.wikipedia.org/wiki/K-sorted_sequence#Algorithms
    //https://en.wikipedia.org/wiki/Partial_sorting
    OrganismsController::precise_sort_high_to_low_dead_organisms_positions(edc);
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {try_make_child(edc, sp, organism, gen);}}
}

void SimulationEngineSingleThread::place_organism(EngineDataContainer &edc, Organism &organism, SimulationParameters &sp) {
    if (edc.record_data) {edc.stc.tbuffer.record_new_organism(organism);}
    for (auto &block: organism.anatomy.organism_blocks) {
        auto pos = block.get_pos(organism.rotation);
        edc.st_grid.get_type(organism.x + pos.x, organism.y + pos.y) = block.type;
        edc.st_grid.get_rotation(organism.x + pos.x, organism.y + pos.y) = get_global_rotation(block.rotation, organism.rotation);
        edc.st_grid.get_organism_index(organism.x + pos.x, organism.y + pos.y) = organism.vector_index;
        if (sp.organisms_destroy_food) {
            auto & num = edc.st_grid.get_food_num(organism.x + pos.x, organism.y + pos.y);
            if (edc.record_data) {edc.stc.tbuffer.record_food_change(organism.x + pos.x, organism.y + pos.y, -num);}
            num = 0;
        }
    }
    if (sp.use_continuous_movement) {
        organism.init_values();
    }
}

void SimulationEngineSingleThread::produce_food(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism, lehmer64 &gen) {
    if (organism.anatomy.c.data[int(BlockTypes::ProducerBlock)-1] == 0) {return;}
    if (organism.anatomy.c.data[int(BlockTypes::MoverBlock)-1] > 0 && !sp.movers_can_produce_food) {return;}
    if (organism.lifetime % sp.produce_food_every_n_life_ticks != 0) {return;}

    if (sp.simplified_food_production) {
        produce_food_simplified(edc, sp, organism, gen, organism.multiplier);
    } else {
        produce_food_complex(edc, sp, organism, gen, organism.multiplier);
    }
}

//just iterate over all producing spaces and try to make food.
void SimulationEngineSingleThread::produce_food_simplified(EngineDataContainer &edc, SimulationParameters &sp,
                                                           Organism &organism, lehmer64 &gen, float multiplier) {
    for (auto & pr: organism.anatomy.producing_space) {
        for (auto &pc: pr) {
            auto x = organism.x + pc.get_pos(organism.rotation).x;
            auto y = organism.y + pc.get_pos(organism.rotation).y;

            auto & type = edc.st_grid.get_type(x, y);

            if (type != BlockTypes::EmptyBlock && type != BlockTypes::FoodBlock) {return;}

            if (std::uniform_real_distribution<float>(0, 1)(gen) < sp.food_production_probability * multiplier) {
                //if couldn't add the food because there is already almost max amount.
                if (!edc.st_grid.add_food_num(x, y, 1, sp.max_food)) {continue;}
                if (edc.record_data) {edc.stc.tbuffer.record_food_change(x, y, 1);}
                if (sp.stop_when_one_food_generated) { return;}
                continue;
            }
        }
    }
}

void SimulationEngineSingleThread::produce_food_complex(EngineDataContainer &edc, SimulationParameters &sp,
                                                        Organism &organism, lehmer64 &gen, float multiplier) {
    for (auto & pr: organism.anatomy.producing_space) {
        //First checks if producer can produce
        if (std::uniform_real_distribution<float>(0, 1)(gen) > sp.food_production_probability * multiplier) { continue;}

        //Then selects one space to produce food.
        auto & pc = pr[std::uniform_int_distribution<int>(0, pr.size()-1)(gen)];

        auto x = organism.x + pc.get_pos(organism.rotation).x;
        auto y = organism.y + pc.get_pos(organism.rotation).y;

        auto & type = edc.st_grid.get_type(x, y);

        if (type != BlockTypes::EmptyBlock && type != BlockTypes::FoodBlock) {return;}

        if (!edc.st_grid.add_food_num(x, y, 1, sp.max_food)) { continue;}
        if (edc.record_data) {edc.stc.tbuffer.record_food_change(x, y, 1);}
        if (sp.stop_when_one_food_generated) { return;}
    }
}

void SimulationEngineSingleThread::eat_food(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism) {
    for (auto & pc: organism.anatomy.eating_space) {
        auto x = organism.x + pc.get_pos(organism.rotation).x;
        auto y = organism.y + pc.get_pos(organism.rotation).y;

        auto & food_num = edc.st_grid.get_food_num(x, y);
        if (food_num >= sp.food_threshold) {
            auto food_eaten = std::min(food_num, sp.food_threshold*pc.num);
            organism.food_collected += food_eaten;
            food_num -= food_eaten;

            if (edc.record_data) {
                edc.stc.tbuffer.record_food_change(x, y, -food_eaten);
            }
        }
    }
}

void SimulationEngineSingleThread::tick_lifetime(EngineDataContainer &edc, Organism &organism, int &i,
                                                 SimulationParameters &sp) {
    organism.lifetime++;
    if (organism.lifetime > organism.max_lifetime || organism.damage > organism.life_points) {
        organism.kill_organism(edc);
        if (edc.record_data) {edc.stc.tbuffer.record_organism_dying(organism.vector_index);}
        for (auto & block: organism.anatomy.organism_blocks) {
            const auto x = organism.x + block.get_pos(organism.rotation).x;
            const auto y = organism.y + block.get_pos(organism.rotation).y;
            edc.st_grid.get_type(x, y) = BlockTypes::EmptyBlock;
            edc.st_grid.get_organism_index(x, y) = -1;
            bool added = edc.st_grid.add_food_num(x, y, block.get_food_cost(*organism.bp), sp.max_food);
            if (edc.record_data && added) {
                edc.stc.tbuffer.record_food_change(x, y, block.get_food_cost(*organism.bp));
            }
        }
    }
}

void SimulationEngineSingleThread::apply_damage(EngineDataContainer & edc, SimulationParameters & sp, Organism &organism) {
    for (auto &block: organism.anatomy.killing_space) {
        const auto x = organism.x + block.get_pos(organism.rotation).x;
        const auto y = organism.y + block.get_pos(organism.rotation).y;
        switch (edc.st_grid.get_type(x, y)) {
            case BlockTypes::EmptyBlock:
            case BlockTypes::FoodBlock:
            case BlockTypes::WallBlock:
            case BlockTypes::ArmorBlock:
                continue;
            default:
                break;
        }
        auto * world_organism = OrganismsController::get_organism_by_index(edc.st_grid.get_organism_index(x, y), edc);
        if (world_organism == nullptr) { continue;}
        if (sp.on_touch_kill) { world_organism->damage = world_organism->life_points + 1; break; }
        world_organism->damage += sp.killer_damage_amount*block.num;
    }
}

void SimulationEngineSingleThread::reserve_observations(std::vector<std::vector<Observation>> &observations,
                                                        std::vector<Organism> &organisms,
                                                        EngineDataContainer &edc) {
    observations.reserve(edc.stc.last_alive_position + 1);
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        auto & organism = edc.stc.organisms[i];
        if (organism.is_dead) { observations.emplace_back(); continue;}

        //if organism has no eyes, movers or is moving, then do not observe.
        if (organism.anatomy.c["eye"] > 0 && organism.anatomy.c["mover"] > 0 && organism.move_counter == 0) {
            observations.emplace_back(std::vector<Observation>(organism.anatomy.c["eye"]));
        } else {
            observations.emplace_back();
        }
    }
}

void SimulationEngineSingleThread::get_observations(EngineDataContainer &edc, SimulationParameters &sp,
                                                    Organism &organism,
                                                    std::vector<std::vector<Observation>> &organism_observations)
                                                    {
    if (organism.anatomy.c["eye"] <= 0 || organism.anatomy.c["mover"] <= 0) {return;}
    if (organism.move_counter != 0) {return;}

    for (int eye_i = 0; eye_i < organism.anatomy.c["eye"]; eye_i++) {
        auto & block = organism.anatomy.eye_block_vec[eye_i];

        auto pos_x = organism.x + block.get_pos(organism.rotation).x;
        auto pos_y = organism.y + block.get_pos(organism.rotation).y;
        // getting global rotation on a simulation grid
        auto block_rotation = block.get_block_rotation_on_grid(organism.rotation);

        auto offset_x = 0;
        auto offset_y = 0;

        switch (block_rotation) {
            case Rotation::UP:    offset_y = -1; break;
            case Rotation::LEFT:  offset_x = -1; break;
            case Rotation::DOWN:  offset_y =  1; break;
            case Rotation::RIGHT: offset_x =  1; break;
        }

#ifdef __DEBUG__
    if ((int)block.rotation < 0 || (int)block.rotation >= 4) {throw std::runtime_error("");}
#endif

        auto last_observation = Observation{BlockTypes::EmptyBlock, 0, block.rotation};

        for (int i = 1; i <= sp.look_range; i++) {
            pos_x += offset_x;
            pos_y += offset_y;

            last_observation.type = edc.st_grid.get_type(pos_x, pos_y);
            last_observation.distance = i;

            if (last_observation.type == BlockTypes::EmptyBlock && edc.st_grid.get_food_num(pos_x, pos_y) >= sp.food_threshold) {
                last_observation.type = BlockTypes::FoodBlock;
            }

            switch (last_observation.type) {
                case BlockTypes::EmptyBlock: continue;
                case BlockTypes::FoodBlock:
                case BlockTypes::WallBlock: goto endfor;
                default:
                    if (!sp.organism_self_blocks_block_sight && edc.st_grid.get_organism_index(pos_x, pos_y) == organism.vector_index) {
                        continue;
                    }
                    goto endfor;
            }
        }
        endfor:
        organism_observations[organism.vector_index][eye_i] = last_observation;
    }
}

void SimulationEngineSingleThread::rotate_organism(EngineDataContainer &edc, Organism &organism, BrainDecision decision,
                                                   SimulationParameters &sp, bool &moved) {
    auto new_int_rotation = static_cast<uint_fast8_t>(organism.rotation);
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
    for (auto & block: organism.anatomy.organism_blocks) {
        if (check_if_block_out_of_bounds(edc, organism, block, new_rotation)) { return;}
        const auto x = organism.x + block.get_pos(new_rotation).x;
        const auto y = organism.y + block.get_pos(new_rotation).y;

        auto type = edc.st_grid.get_type(x, y);
        auto organism_index = edc.st_grid.get_organism_index(x, y);

        if (sp.food_blocks_movement) {
            auto food_num = edc.st_grid.get_food_num(x, y);
            if ((type != BlockTypes::EmptyBlock && organism_index != organism.vector_index) || (type == BlockTypes::EmptyBlock && food_num >= sp.food_threshold)) {return;}
        } else {
            if (type != BlockTypes::EmptyBlock && organism_index != organism.vector_index) {return;}
        }
    }

    for (auto & block: organism.anatomy.organism_blocks) {
        const auto pos = block.get_pos(organism.rotation);
        const auto x = organism.x + pos.x; const auto y = organism.y + pos.y;
        edc.st_grid.get_type(x, y) = BlockTypes::EmptyBlock;
        edc.st_grid.get_organism_index(x, y) = -1;
    }

    //If there is a place for rotated organism, then rotation can happen
    organism.rotation = new_rotation;
    for (auto & block: organism.anatomy.organism_blocks) {
        const auto pos = block.get_pos(organism.rotation);
        const auto x = organism.x + pos.x; const auto y = organism.y + pos.y;
        edc.st_grid.get_type(x, y) = block.type;
        edc.st_grid.get_rotation(x, y) = get_global_rotation(block.rotation, organism.rotation);
        edc.st_grid.get_organism_index(x, y) = organism.vector_index;
        if (sp.organisms_destroy_food) { edc.st_grid.get_food_num(x, y) = 0;}
    }
}

void SimulationEngineSingleThread::move_organism(EngineDataContainer &edc, Organism &organism, BrainDecision decision,
                                                 SimulationParameters &sp, bool &moved) {
    int new_x = -1;
    int new_y = -1;

    if (sp.use_continuous_movement) {
        if (!calculate_continuous_move(edc, organism, sp, new_x, new_y)) { return;}
    } else {
        if (!calculate_discrete_movement(edc, organism, decision, sp, new_x, new_y)) { return; }
    }

    for (auto & block: organism.anatomy.organism_blocks) {
        auto pos = block.get_pos(organism.rotation);
        edc.st_grid.get_type(organism.x + pos.x, organism.y + pos.y) = BlockTypes::EmptyBlock;
        edc.st_grid.get_organism_index(organism.x + pos.x, organism.y + pos.y) = -1;
    }

    for (auto & block: organism.anatomy.organism_blocks) {
        auto pos = block.get_pos(organism.rotation);
        edc.st_grid.get_type(new_x + pos.x, new_y + pos.y) = block.type;
        edc.st_grid.get_rotation(new_x + pos.x, new_y + pos.y) = get_global_rotation(block.rotation, organism.rotation);
        edc.st_grid.get_organism_index(new_x + pos.x, new_y + pos.y) = organism.vector_index;
        if (sp.organisms_destroy_food) { edc.st_grid.get_food_num(new_x + pos.x, new_y + pos.y) = 0;}
    }

    organism.x = new_x;
    organism.y = new_y;
    moved = true;
}

bool SimulationEngineSingleThread::calculate_continuous_move(EngineDataContainer &edc, Organism &organism,
                                                             const SimulationParameters &sp, int &new_x,
                                                             int &new_y) {
    float mass = organism.food_needed; // + organism.food_collected
    auto & cd = organism.cdata;
    cd.p_vx += (cd.p_fx - cd.p_vx * sp.continuous_movement_drag) / mass * organism.anatomy.c["mover"];
    cd.p_vy += (cd.p_fy - cd.p_vy * sp.continuous_movement_drag) / mass * organism.anatomy.c["mover"];

    cd.p_x += cd.p_vx;
    cd.p_y += cd.p_vy;

    cd.p_fx = 0;
    cd.p_fy = 0;

    new_x = std::round(cd.p_x);
    new_y = std::round(cd.p_y);

    if (new_x != organism.x || new_y != organism.y) {
        bool is_clear = true;

        for (auto & block: organism.anatomy.organism_blocks) {
            auto pos = block.get_pos(organism.rotation);

            auto new_pos = Vector2{new_x + pos.x, new_y + pos.y};
            //organism can get enough velocity to have new coordinate reach over wall boundary
            if (check_if_out_of_bounds(edc, new_pos.x, new_pos.y)) {is_clear = false; break;}

            auto type = edc.st_grid.get_type(new_pos.x, new_pos.y);
            auto organism_index = edc.st_grid.get_organism_index(new_x + pos.x, new_y + pos.y);

            if (sp.food_blocks_movement) {
                auto food_num = edc.st_grid.get_food_num(new_x + pos.x, new_y + pos.y);
                if ((type != BlockTypes::EmptyBlock && organism_index != organism.vector_index)
                 || (type == BlockTypes::EmptyBlock && food_num >= sp.food_threshold)) {is_clear = false; break;}
            } else {
                if ( type != BlockTypes::EmptyBlock && organism_index != organism.vector_index) {is_clear = false; break;}
            }
        }

        if (!is_clear) {
            cd.p_x = organism.x;
            cd.p_y = organism.y;
            cd.p_vx = 0;
            cd.p_vy = 0;
            return false;
        }
        return true;
    }
    return false;
}

bool SimulationEngineSingleThread::calculate_discrete_movement(EngineDataContainer &edc, Organism &organism,
                                                               BrainDecision decision,
                                                               const SimulationParameters &sp, int &new_x,
                                                               int &new_y) {
    new_x = organism.x;
    new_y = organism.y;
    switch (decision) {
        case BrainDecision::MoveUp:    new_y -= 1; break;
        case BrainDecision::MoveLeft:  new_x -= 1; break;
        case BrainDecision::MoveDown:  new_y += 1; break;
        case BrainDecision::MoveRight: new_x += 1; break;
        default: break;
    }

    //Organism can move only by 1 block a simulation tick, so it will be stopped by a wall and doesn't need an out-of-bounds check.
    for (auto & block: organism.anatomy.organism_blocks) {
        auto pos = block.get_pos(organism.rotation);

        auto type = edc.st_grid.get_type(new_x + pos.x, new_y + pos.y);
        auto organism_index = edc.st_grid.get_organism_index(new_x + pos.x, new_y + pos.y);

        if (sp.food_blocks_movement) {
            auto food_num = edc.st_grid.get_food_num(new_x + pos.x, new_y + pos.y);
            if ((type != BlockTypes::EmptyBlock && organism_index != organism.vector_index)
             || (type == BlockTypes::EmptyBlock && food_num >= sp.food_threshold)) {return false;}
        } else {
            if ( type != BlockTypes::EmptyBlock && organism_index != organism.vector_index) {return false;}
        }
    }
    return true;
}

void SimulationEngineSingleThread::make_decision(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism, lehmer64 &gen) {
    bool moved = false;

    if (sp.use_continuous_movement) {
        if (organism.anatomy.c["mover"] > 0) {
            move_organism(edc, organism, organism.last_decision_observation.decision, sp, moved);
            organism.move_counter++;
        }
    } else {
        switch (organism.last_decision_observation.decision) {
            case BrainDecision::MoveUp:
            case BrainDecision::MoveDown:
            case BrainDecision::MoveLeft:
            case BrainDecision::MoveRight:
                if (organism.anatomy.c["mover"] > 0) {
                    move_organism(edc, organism, organism.last_decision_observation.decision, sp, moved);
                    organism.move_counter++;
                }
                break;
            default:
                break;
        }
    }

    if ((!sp.set_fixed_move_range && organism.move_counter >= organism.move_range) || (sp.set_fixed_move_range && sp.min_move_range <= organism.move_counter)) {
        organism.move_counter = 0;
    }
    if ((organism.move_counter == 0 || sp.rotate_every_move_tick) && organism.anatomy.c["mover"] > 0 && sp.runtime_rotation_enabled) {
        if (organism.last_decision_observation.decision != organism.last_decision || sp.no_random_decisions || sp.use_continuous_movement) {
            organism.last_decision = organism.last_decision_observation.decision;
            rotate_organism(edc, organism, static_cast<BrainDecision>(std::uniform_int_distribution<int>(4, 6)(gen)),
                            sp, moved);
        }
    }

    if (edc.record_data && moved) {edc.stc.tbuffer.record_organism_move_change(organism.vector_index, organism.x, organism.y, organism.rotation);}
}

void SimulationEngineSingleThread::try_make_child(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism,
                                                  lehmer64 &gen) {
    if (organism.child_pattern_index < 0) { organism.child_pattern_index = organism.create_child(gen, edc);}
    // if max_organisms < 0, then unlimited.
    if (edc.max_organisms >= 0 && edc.stc.num_alive_organisms >= edc.max_organisms) {return;}
    auto child_food_needed = OrganismsController::get_child_organism_by_index(organism.child_pattern_index, edc)->food_needed;
    if (organism.food_collected < child_food_needed) {return;}
    if (sp.failed_reproduction_eats_food) {organism.food_collected -= child_food_needed;}
    place_child(edc, sp, organism, gen);
}

void SimulationEngineSingleThread::place_child(EngineDataContainer &edc, SimulationParameters &sp, Organism &organism,
                                               lehmer64 &gen) {
    auto to_place = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(gen));
    Rotation rotation;
    if (sp.reproduction_rotation_enabled) {
        rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(gen));
    } else {
        rotation = Rotation::UP;
    }

    int distance = sp.min_reproducing_distance;
    if (!sp.reproduction_distance_fixed) {
        distance = std::uniform_int_distribution<int>(sp.min_reproducing_distance, sp.max_reproducing_distance)(gen);
    }

    auto * child_pattern = OrganismsController::get_child_organism_by_index(organism.child_pattern_index, edc);
    child_pattern->anatomy.recenter_blocks(sp.recenter_to_imaginary_pos);

    child_pattern->rotation = rotation;

    child_pos_calculator(organism, to_place, distance, edc);

    if (sp.check_if_path_is_clear) {
        int c_distance = std::abs(organism.x - child_pattern->x) + std::abs(organism.y - child_pattern->y);

        for (auto & block: child_pattern->anatomy.organism_blocks) {
            if (!path_is_clear(organism.x + block.get_pos(rotation).x,
                               organism.y + block.get_pos(rotation).y,
                               to_place,
                               c_distance,
                               organism.vector_index, edc, sp)) {
                return;}
        }
    }

    //checking, if there is space for a child
    for (auto & block: child_pattern->anatomy.organism_blocks) {
        if (check_if_block_out_of_bounds(edc, *child_pattern, block, child_pattern->rotation)) {return;}

        auto type = edc.st_grid.get_type(child_pattern->x + block.get_pos(child_pattern->rotation).x,
                                           child_pattern->y + block.get_pos(child_pattern->rotation).y);
        if (sp.food_blocks_reproduction) {
            auto food_num = edc.st_grid.get_food_num(child_pattern->x + block.get_pos(child_pattern->rotation).x,
                                                     child_pattern->y + block.get_pos(child_pattern->rotation).y);
            if (type != BlockTypes::EmptyBlock || food_num >= sp.food_threshold) {return;}
        } else {
            if (type != BlockTypes::EmptyBlock) {return;}
        }
    }

    //only deduct food when reproduction is successful and flag is false
    if (!sp.failed_reproduction_eats_food) {
        organism.food_collected -= child_pattern->food_needed;
    }

    //It is needed because pointer to organism can become invalid during emplace_child_organism_to_main_vector if organism vector resizes.
    auto main_organism_index = organism.vector_index;
    auto child_index = OrganismsController::emplace_child_organisms_to_main_vector(child_pattern, edc);
    place_organism(edc, *OrganismsController::get_organism_by_index(child_index, edc), sp);
    if (edc.record_data) {edc.stc.tbuffer.record_new_organism(*OrganismsController::get_organism_by_index(child_index, edc));}
    OrganismsController::get_organism_by_index(main_organism_index, edc)->child_pattern_index = -1;
}

void SimulationEngineSingleThread::child_pos_calculator(Organism &organism, const Rotation to_place, int distance,
                                                        EngineDataContainer &edc) {
    auto o_min_y = 0;
    auto o_min_x = 0;
    auto o_max_y = 0;
    auto o_max_x = 0;

    auto c_min_y = 0;
    auto c_min_x = 0;
    auto c_max_y = 0;
    auto c_max_x = 0;

    auto * child_pattern = OrganismsController::get_child_organism_by_index(organism.child_pattern_index, edc);

    //To get the position of a child on an axis you need to know the largest/smallest position of organism blocks on an axis
    // and smallest/largest position of a child organism block + -1/+1 to compensate for organism center and + -/+ distance.
    //For example, if a child is reproduced up of organism, you need the upmost coordinate among organism blocks and
    // lowest coordinate of child organism blocks with -1 to compensate for organism center. With that, child will spawn
    // basically attached to the parent organism, so then you subtract the random distance.

    switch (to_place) {
        case Rotation::UP:
            o_min_y = INT32_MAX;
            c_max_y = INT32_MIN;
            for (auto & block: organism.anatomy.organism_blocks) {
                auto pos = block.get_pos(organism.rotation);
                if (pos.y < o_min_y) { o_min_y = pos.y;}
            }
            for (auto & block: child_pattern->anatomy.organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.y > c_max_y) { c_max_y = pos.y;}
            }
            o_min_y -= distance;
            c_max_y = -abs(c_max_y) - 1;
            break;
        case Rotation::LEFT:
            o_min_x = INT32_MAX;
            c_max_x = INT32_MIN;
            for (auto & block: organism.anatomy.organism_blocks) {
                auto pos = block.get_pos(organism.rotation);
                if (pos.x < o_min_x) { o_min_x = pos.x;}
            }
            for (auto & block: child_pattern->anatomy.organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.x > c_max_x) { c_max_x = pos.x;}
            }
            o_min_x -= distance;
            c_max_x = -abs(c_max_x) - 1;
            break;
        case Rotation::DOWN:
            o_max_y = INT32_MIN;
            c_min_y = INT32_MAX;
            for (auto & block: organism.anatomy.organism_blocks) {
                auto pos = block.get_pos(organism.rotation);
                if (pos.y > o_max_y) { o_max_y = pos.y;}
            }
            for (auto & block: child_pattern->anatomy.organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.y < c_min_y) { c_min_y = pos.y;}
            }
            o_max_y += distance;
            c_min_y = abs(c_min_y) + 1;
            break;
        case Rotation::RIGHT:
            o_max_x = INT32_MIN;
            c_min_x = INT32_MAX;
            for (auto & block: organism.anatomy.organism_blocks){
                auto pos = block.get_pos(organism.rotation);
                if (pos.x > o_max_x) { o_max_x = pos.x;}
            }
            for (auto & block: child_pattern->anatomy.organism_blocks) {
                auto pos = block.get_pos(child_pattern->rotation);
                if (pos.x < c_min_x) { c_min_x = pos.x;}
            }
            o_max_x += distance;
            c_min_x = abs(c_min_x) + 1;
            break;
    }

    child_pattern->x = organism.x + o_min_x + o_max_x + c_min_x + c_max_x;
    child_pattern->y = organism.y + o_min_y + o_max_y + c_min_y + c_max_y;
}

bool SimulationEngineSingleThread::check_if_out_of_bounds(EngineDataContainer &edc, int x, int y) {
    return (x < 0 ||
            x > edc.simulation_width - 1 ||
            y < 0 ||
            y > edc.simulation_height - 1);
}

// if any is true, then check fails, else check succeeds
bool SimulationEngineSingleThread::check_if_block_out_of_bounds(EngineDataContainer &edc, Organism &organism,
                                                                BaseSerializedContainer &block, Rotation rotation) {
    return check_if_out_of_bounds(edc,
                                  organism.x + block.get_pos(rotation).x,
                                  organism.y + block.get_pos(rotation).y);
}

bool SimulationEngineSingleThread::path_is_clear(int x, int y, Rotation direction, int steps, int32_t allow_organism, EngineDataContainer &edc,
                   SimulationParameters &sp) {
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
        if (check_if_out_of_bounds(edc, x, y)) {
            return false;
        }

        auto type = edc.st_grid.get_type(x, y);
        auto organism_index = edc.st_grid.get_organism_index(x, y);
        if (type != BlockTypes::EmptyBlock && organism_index != allow_organism) {return false;}
    }
    return true;
}

std::array<int, 4> SimulationEngineSingleThread::get_organism_dimensions(Organism &organism) {
    std::array<int, 4> output{INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN};
    auto & anatomy = organism.anatomy;

    for (auto & block: anatomy.organism_blocks) {
        if (block.relative_x < output[0]) {output[0] = block.relative_x;}
        if (block.relative_y < output[1]) {output[1] = block.relative_y;}
        if (block.relative_x > output[2]) {output[2] = block.relative_x;}
        if (block.relative_y > output[3]) {output[3] = block.relative_y;}
    }

    return output;
}
