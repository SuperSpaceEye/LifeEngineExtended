//
// Created by spaceeye on 16.05.2022.
//

#include "../SimulationEngine.h"
#include "SimulationEngineSingleThread.h"

void SimulationEngineSingleThread::single_threaded_tick(EngineDataContainer * dc, SimulationParameters * sp, boost::mt19937 *mt) {
    for (auto & organism: dc->to_place_organisms) {place_organism(dc, organism); dc->organisms.emplace_back(organism);}
    dc->to_place_organisms.clear();

    auto to_erase = std::vector<int>{};

    for (auto & organism: dc->organisms) {eat_food(dc, sp, organism);}
    for (auto & organism: dc->organisms) {produce_food(dc, sp, organism, *mt);}

    if (sp->killer_damage_amount > 0) {
        for (auto &organism: dc->organisms) { apply_damage(dc, sp, organism); }
    }

    for (int i = 0; i < dc->organisms.size(); i++)  {tick_lifetime(dc, to_erase, dc->organisms[i], i);}
    for (int i = 0; i < to_erase.size(); ++i)       {erase_organisms(dc, to_erase, i);}

    auto organisms_observations = std::vector<std::vector<Observation>>();
    reserve_observations(organisms_observations, dc->organisms, sp);
    get_observations(dc, sp, dc->organisms, organisms_observations);

    for (int i = 0; i < dc->organisms.size(); i++)  {think_decision(dc, sp, dc->organisms[i], organisms_observations[i], mt);}
    for (int i = 0; i < dc->organisms.size(); i++)  {make_decision(dc, sp, dc->organisms[i]);}

    for (auto & organism: dc->organisms) {try_make_child(dc, sp, organism, dc->to_place_organisms, mt);}
}

void SimulationEngineSingleThread::place_organism(EngineDataContainer *dc, Organism *organism) {
    for (auto &block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].type = block.type;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].neighbors = block.get_rotated_block_neighbors(organism->rotation);
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].rotation = block.rotation;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].organism = organism;
    }
}

//Each producer will add one run of producing a food
void SimulationEngineSingleThread::produce_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, boost::mt19937 &mt) {
    if (organism->organism_anatomy->_producer_blocks <= 0) {return;}
    if (organism->organism_anatomy->_mover_blocks > 0 && !sp->movers_can_produce_food) {return;}
    if (organism->lifetime % sp->produce_food_every_n_life_ticks != 0) {return;}

    for (int i = 0; i < organism->organism_anatomy->_producer_blocks; i++) {
        for (auto & pc: organism->organism_anatomy->_producing_space) {
            //if (check_if_out_of_bounds(dc, organism, pc)) {continue;}
            if (dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type == EmptyBlock) {
                if (std::uniform_real_distribution<float>(0, 1)(mt) < sp->food_production_probability) {
                    dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type = FoodBlock;
                    break;
                }
            }
        }
    }
}

void SimulationEngineSingleThread::eat_food(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto & pc: organism->organism_anatomy->_eating_space) {
        if (dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type == FoodBlock) {
            dc->CPU_simulation_grid[organism->x + pc.get_pos(organism->rotation).x][organism->y + pc.get_pos(organism->rotation).y].type = EmptyBlock;
            organism->food_collected++;
        }
    }
}

void SimulationEngineSingleThread::tick_lifetime(EngineDataContainer *dc, std::vector<int>& to_erase, Organism *organism, int organism_pos) {
    organism->lifetime++;
    if (organism->lifetime > organism->max_lifetime || organism->damage > organism->life_points) {
        for (auto & block: organism->organism_anatomy->_organism_blocks) {
            dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y].type = FoodBlock;
            dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y].organism = nullptr;
        }
        to_erase.push_back(organism_pos);
    }
}

void SimulationEngineSingleThread::erase_organisms(EngineDataContainer *dc, std::vector<int> &to_erase, int i) {
    //when erasing organism vector will decrease, so we must account for that
    delete dc->organisms[to_erase[i]-i];
    dc->organisms.erase(dc->organisms.begin() + to_erase[i] - i);
}

void SimulationEngineSingleThread::apply_damage(EngineDataContainer * dc, SimulationParameters * sp, Organism *organism) {
    for (auto & block: organism->organism_anatomy->_single_adjacent_space) {
        if (block.is_armored) {continue;}
        if (dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x][organism->y + block.get_pos(organism->rotation).y].type == KillerBlock) {
            if (sp->on_touch_kill) { organism->damage = organism->life_points + 1; break;}
            organism->damage += sp->killer_damage_amount;
        }
    }
}

void SimulationEngineSingleThread::reserve_observations(std::vector<std::vector<Observation>> &observations,
                                                        std::vector<Organism *> &organisms,
                                                        SimulationParameters *sp) {
    auto observations_count = std::vector<int>{};
    observations_count.reserve(organisms.size());
    for (auto & organism: organisms) {
        // if organism is moving, then do not observe.
        if (organism->move_counter == 0 && organism->organism_anatomy->_mover_blocks > 0 && organism->organism_anatomy->_eye_blocks > 0) {
            observations_count.emplace_back(organism->organism_anatomy->_eye_blocks);
        } else {
            observations_count.emplace_back(0);
        }
    }
    observations.reserve(organisms.size());
    for (auto & item: observations_count) {observations.emplace_back(std::vector<Observation>(item));}
}

void SimulationEngineSingleThread::get_observations(EngineDataContainer *dc, SimulationParameters *sp,
                                                    std::vector<Organism *> &organisms,
                                                    std::vector<std::vector<Observation>> &organism_observations) {
    auto organism_i = -1;
    for (auto & organism : organisms) {
        organism_i++;
        if (organism->organism_anatomy->_eye_blocks <= 0 || organism->organism_anatomy->_mover_blocks <=0) {continue;}
        if (organism->move_counter != 0) {continue;}
        auto eye_i = -1;
        for (auto & block: organism->organism_anatomy->_organism_blocks) {
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

            auto last_observation = Observation{EmptyBlock, 0, block.rotation};

            for (int i = 1; i < sp->look_range; i++) {
                pos_x += offset_x;
                pos_y += offset_y;

                if (check_if_out_of_bounds(dc, pos_x, pos_y)) {break;}

                last_observation.type = dc->CPU_simulation_grid[pos_x][pos_y].type;
                last_observation.distance = i;

                if (last_observation.type == BlockTypes::WallBlock) {break;}
                if (last_observation.type == BlockTypes::FoodBlock) {break;}
                if (last_observation.type != BlockTypes::EmptyBlock) {
                    if (!sp->organism_self_blocks_block_sight) {
                        if(dc->CPU_simulation_grid[pos_x][pos_y].organism == organism) {
                            continue;
                        }
                    }
                    break;
                }
            }
            organism_observations[organism_i][eye_i] = last_observation;
        }
    }
}

void SimulationEngineSingleThread::rotate_organism(EngineDataContainer * dc, Organism *organism, BrainDecision decision) {
    auto new_int_rotation = static_cast<int>(organism->rotation);
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
    if (new_int_rotation < 0) {new_int_rotation+=4;}
    if (new_int_rotation > 3) {new_int_rotation-=4;}

    auto new_rotation = static_cast<Rotation>(new_int_rotation);

    //checks if space for organism is empty, or contains itself
    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        if (check_if_block_out_of_bounds(dc, organism, block, new_rotation) ||
                (dc->CPU_simulation_grid[organism->x + block.get_pos(new_rotation).x]
                               [organism->y + block.get_pos(new_rotation).y].type != EmptyBlock &&
                dc->CPU_simulation_grid[organism->x + block.get_pos(new_rotation).x]
                               [organism->y + block.get_pos(new_rotation).y].organism != organism)) {
            return;
        }
    }

    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].type = BlockTypes::EmptyBlock;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].organism = nullptr;
    }

    //If there is a place for rotated organism, then rotation can happen
    organism->rotation = new_rotation;
    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].type = block.type;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].rotation = block.rotation;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].neighbors = block.get_rotated_block_neighbors(organism->rotation);
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].organism = organism;
    }
}

void SimulationEngineSingleThread::move_organism(EngineDataContainer * dc, Organism *organism, BrainDecision decision) {
    // rotates movement relative to simulation grid
    auto new_int_decision = static_cast<int>(decision) + static_cast<int>(organism->rotation);
    if (new_int_decision > 3) {new_int_decision -= 4;}
    auto new_decision = static_cast<BrainDecision>(new_int_decision);

    int new_x = organism->x;
    int new_y = organism->y;

    switch (new_decision) {
        case BrainDecision::MoveUp:
            new_y -= 1;
            break;
        case BrainDecision::MoveLeft:
            new_x -= 1;
            break;
        case BrainDecision::MoveDown:
            new_y += 1;
            break;
        case BrainDecision::MoveRight:
            new_x += 1;
            break;
        default: break;
    }

    //Organism can move only by 1 block a simulation tick, so it will be stopped by a wall and doesn't need an out-of-bounds check.
    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        if (dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
                               [new_y + block.get_pos(organism->rotation).y].type != EmptyBlock &&
                dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
                [new_y + block.get_pos(organism->rotation).y].organism != organism) {
            return;
        }
    }

    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].type = BlockTypes::EmptyBlock;
        dc->CPU_simulation_grid[organism->x + block.get_pos(organism->rotation).x]
        [organism->y + block.get_pos(organism->rotation).y].organism = nullptr;
    }

    for (auto & block: organism->organism_anatomy->_organism_blocks) {
        dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
        [new_y + block.get_pos(organism->rotation).y].type = block.type;
        dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
        [new_y + block.get_pos(organism->rotation).y].rotation = block.rotation;
        dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
        [new_y + block.get_pos(organism->rotation).y].neighbors = block.get_rotated_block_neighbors(organism->rotation);
        dc->CPU_simulation_grid[new_x + block.get_pos(organism->rotation).x]
        [new_y + block.get_pos(organism->rotation).y].organism = organism;
    }

    organism->x = new_x;
    organism->y = new_y;
}

void SimulationEngineSingleThread::think_decision(EngineDataContainer *dc,
                                                  SimulationParameters *sp,
                                                  Organism *organism,
                                                  std::vector<Observation> &organism_observations,
                                                  boost::mt19937 *mt) {
    if (organism->move_counter == 0) {
        organism->last_decision = organism->brain->get_decision(organism_observations, *mt);
    }
}

void SimulationEngineSingleThread::make_decision(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism) {
    switch (organism->last_decision) {
        case BrainDecision::MoveUp:
        case BrainDecision::MoveDown:
        case BrainDecision::MoveLeft:
        case BrainDecision::MoveRight:
            if (organism->organism_anatomy->_mover_blocks > 0) {
                move_organism(dc, organism, organism->last_decision);
                organism->move_counter++;
            }
            break;
        case BrainDecision::RotateLeft:
        case BrainDecision::RotateRight:
        case BrainDecision::Flip:
            if (organism->organism_anatomy->_mover_blocks > 0 && sp->runtime_rotation_enabled) {
                rotate_organism(dc, organism, organism->last_decision);
            }
            break;

        default: break;
    }
    if ((organism->move_counter >= organism->move_range) || (sp->set_fixed_move_range && sp->min_move_range == organism->move_counter)) {
        organism->move_counter = 0;
    }
}

void SimulationEngineSingleThread::try_make_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                                  std::vector<Organism *> &child_organisms, boost::mt19937 *mt) {
    if (!organism->child_ready) {organism->child_pattern = organism->create_child(mt); organism->child_ready=true;}
    // if max_organisms < 0, then unlimited.
    if (dc->max_organisms >= 0 && dc->organisms.size() + dc->to_place_organisms.size() >= dc->max_organisms) {return;}
    if (organism->food_collected >= organism->child_pattern->food_needed) {
        if (sp->failed_reproduction_eats_food) {organism->food_collected -= organism->child_pattern->food_needed;}
        place_child(dc, sp, organism, child_organisms, mt);}
}

void SimulationEngineSingleThread::place_child(EngineDataContainer *dc, SimulationParameters *sp, Organism *organism,
                                               std::vector<Organism *> &child_organisms, boost::mt19937 *mt) {
    auto to_place = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
    Rotation rotation;
    if (sp->reproduction_rotation_enabled) {
        rotation = static_cast<Rotation>(std::uniform_int_distribution<int>(0, 3)(*mt));
    } else {
        rotation = Rotation::UP;
    }

    int distance = sp->min_reproducing_distance;
    if (!sp->reproduction_distance_fixed) {
        distance = std::uniform_int_distribution<int>(sp->min_reproducing_distance, sp->max_reproducing_distance)(*mt);
    }
    //UP - min_y,
    //LEFT - min_x
    //DOWN - max_y
    //RIGHT - max_x

    auto min_y = 0;
    auto min_x = 0;
    auto max_y = 0;
    auto max_x = 0;


    //a width and height of an organism can only change by one, so to be safe, the distance between organisms = size_of_base_organism + 1
    switch (to_place) {
        case Rotation::UP:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).y < min_y) {min_y = block.get_pos(organism->rotation).y;}
            }
            min_y -= distance;
//            min_x = 0;
//            max_y = 0;
//            max_x = 0;
            break;
        case Rotation::LEFT:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).x < min_x) {min_x = block.get_pos(organism->rotation).x;}
            }
//            min_y = 0;
            min_x -= distance;
//            max_y = 0;
//            max_x = 0;
            break;
        case Rotation::DOWN:
            for (auto & block: organism->organism_anatomy->_organism_blocks) {
                if (block.get_pos(organism->rotation).y > max_y) {max_y = block.get_pos(organism->rotation).y;}
            }
//            min_y = 0;
//            min_x = 0;
            max_y += distance;
//            max_x = 0;
            break;
        case Rotation::RIGHT:
            for (auto & block: organism->organism_anatomy->_organism_blocks){
                if (block.get_pos(organism->rotation).x > max_x) {max_x = block.get_pos(organism->rotation).x;}
            }
//            min_y = 0;
//            min_x = 0;
//            max_y = 0;
            max_x += distance;
            break;
    }

    organism->child_pattern->x = organism->x + min_x + max_x;
    organism->child_pattern->y = organism->y + min_y + max_y;
    organism->child_pattern->rotation = rotation;

    //checking, if there is space for a child
    for (auto & block: organism->child_pattern->organism_anatomy->_organism_blocks) {
        if (check_if_block_out_of_bounds(dc, organism->child_pattern, block, organism->child_pattern->rotation)) {return;}

        if (dc->CPU_simulation_grid[organism->child_pattern->x + block.get_pos(organism->child_pattern->rotation).x]
                               [organism->child_pattern->y + block.get_pos(organism->child_pattern->rotation).y].type != EmptyBlock)
        {return;}

//        if (dc->CPU_simulation_grid[organism->child_pattern->x + block.get_pos(organism->child_pattern->rotation).x]
//                [organism->child_pattern->y + block.get_pos(organism->child_pattern->rotation).y].type == BlockTypes::EmptyBlock ||
//            (!sp->food_blocks_reproduction && dc->CPU_simulation_grid[organism->child_pattern->x + block.get_pos(organism->child_pattern->rotation).x]
//                [organism->child_pattern->y + block.get_pos(organism->child_pattern->rotation).y].type == BlockTypes::FoodBlock))
//        {continue;}
//        return;
    }

    child_organisms.emplace_back(organism->child_pattern);
    //will happen only when reproduction is successful
    if (!sp->failed_reproduction_eats_food) {
        organism->food_collected -= organism->child_pattern->food_needed;
    }
    organism->child_pattern = nullptr;
    organism->child_ready = false;
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
