// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

#include <utility>

#include "../../SimulationEngine/OrganismsController.h"

Organism::Organism(int x, int y, Rotation rotation, Anatomy anatomy, Brain brain, OrganismConstructionCode occ,
                   SimulationParameters *sp, OrganismBlockParameters *block_parameters, OCCParameters *occp,
                   OCCLogicContainer *occl, int move_range, float anatomy_mutation_rate, float brain_mutation_rate) :
        anatomy(anatomy), sp(sp), bp(block_parameters), brain(brain), occ(occ), occp(occp), occl(occl),
        OrganismData(x,
                                y,
                                rotation,
                                move_range,
                                anatomy_mutation_rate,
                                brain_mutation_rate) {
    init_values();
}

Organism::Organism(Organism *organism): anatomy(organism->anatomy), sp(organism->sp), occp(organism->occp), occl(organism->occl),
                                        bp(organism->bp), brain(organism->brain), occ(organism->occ),
                                        OrganismData(organism->x,
                                                     organism->y,
                                                     organism->rotation,
                                                     organism->move_range,
                                                     organism->anatomy_mutation_rate,
                                                     organism->brain_mutation_rate) {
    init_values();
}

void Organism::init_values() {
    calculate_max_life();
    calculate_organism_lifetime();
    calculate_food_needed();
    auto vec = anatomy.recenter_blocks(sp->recenter_to_imaginary_pos);

    //just to reuse rotation of positions logic
    auto temp = BaseSerializedContainer{vec.x, vec.y};
    //because organism can be rotated, the shift positions on grid also need to be rotated.
    x += temp.get_pos(rotation).x;
    y += temp.get_pos(rotation).y;

    multiplier = 1;

    if (sp->multiply_food_production_prob) {
        multiplier *= anatomy._producer_blocks;
    }
}

//TODO it can be made more efficiently, but i want (in the future) mutate block parameters individually.
float Organism::calculate_max_life() {
    life_points = 0;
    for (auto& item: anatomy._organism_blocks) {
        switch (item.type) {
            case BlockTypes::MouthBlock:    life_points += bp->MouthBlock.   life_point_amount; break;
            case BlockTypes::ProducerBlock: life_points += bp->ProducerBlock.life_point_amount; break;
            case BlockTypes::MoverBlock:    life_points += bp->MoverBlock.   life_point_amount; break;
            case BlockTypes::KillerBlock:   life_points += bp->KillerBlock.  life_point_amount; break;
            case BlockTypes::ArmorBlock:    life_points += bp->ArmorBlock.   life_point_amount; break;
            case BlockTypes::EyeBlock:      life_points += bp->EyeBlock.     life_point_amount; break;
            default: throw std::runtime_error("Unknown block");
        }
    }
    return life_points;
}

int Organism::calculate_organism_lifetime() {
    float lifetime_weights = 0;
    for (auto & block: anatomy._organism_blocks) {
        switch (block.type) {
            case BlockTypes::MouthBlock:    lifetime_weights += bp->MouthBlock.   lifetime_weight; break;
            case BlockTypes::ProducerBlock: lifetime_weights += bp->ProducerBlock.lifetime_weight; break;
            case BlockTypes::MoverBlock:    lifetime_weights += bp->MoverBlock.   lifetime_weight; break;
            case BlockTypes::KillerBlock:   lifetime_weights += bp->KillerBlock.  lifetime_weight; break;
            case BlockTypes::ArmorBlock:    lifetime_weights += bp->ArmorBlock.   lifetime_weight; break;
            case BlockTypes::EyeBlock:      lifetime_weights += bp->EyeBlock.     lifetime_weight; break;
            default: throw std::runtime_error("Unknown block");
        }
    }
    max_lifetime = static_cast<int>(lifetime_weights * sp->lifespan_multiplier);
    return max_lifetime;
}

float Organism::calculate_food_needed() {
    food_needed = sp->extra_reproduction_cost + sp->extra_mover_reproductive_cost * (anatomy._mover_blocks > 0);
    for (auto & block: anatomy._organism_blocks) {
        switch (block.type) {
            case BlockTypes::MouthBlock:    food_needed += bp->MouthBlock.   food_cost_modifier; break;
            case BlockTypes::ProducerBlock: food_needed += bp->ProducerBlock.food_cost_modifier; break;
            case BlockTypes::MoverBlock:    food_needed += bp->MoverBlock.   food_cost_modifier; break;
            case BlockTypes::KillerBlock:   food_needed += bp->KillerBlock.  food_cost_modifier; break;
            case BlockTypes::ArmorBlock:    food_needed += bp->ArmorBlock.   food_cost_modifier; break;
            case BlockTypes::EyeBlock:      food_needed += bp->EyeBlock.     food_cost_modifier; break;
            default: throw std::runtime_error("Unknown block");
        }
    }
    return food_needed;
}

void Organism::mutate_anatomy(Anatomy &new_anatomy, float &_anatomy_mutation_rate, lehmer64 *gen,
                              OrganismConstructionCode &new_occ) {
    bool mutate_anatomy;
    _anatomy_mutation_rate = anatomy_mutation_rate;

    if (sp->use_anatomy_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(*gen) <= sp->anatomy_mutation_rate_delimiter) {
            _anatomy_mutation_rate += sp->anatomy_mutations_rate_mutation_step;
            if (_anatomy_mutation_rate > 1) {_anatomy_mutation_rate = 1;}
        } else {
            _anatomy_mutation_rate -= sp->anatomy_mutations_rate_mutation_step;
            if (_anatomy_mutation_rate < sp->anatomy_min_possible_mutation_rate) {
                _anatomy_mutation_rate = sp->anatomy_min_possible_mutation_rate;
            }
        }
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(*gen) <= _anatomy_mutation_rate;
    } else {
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(*gen) <= sp->global_anatomy_mutation_rate;
    }

    if (mutate_anatomy) {
        if (!sp->use_occ) {
            int total_chance = 0;
            total_chance += sp->add_cell;
            total_chance += sp->change_cell;
            total_chance += sp->remove_cell;

            int choice = std::uniform_int_distribution<int>(0, total_chance)(*gen);

            if (choice < sp->add_cell) {new_anatomy = Anatomy(anatomy.add_random_block(*bp, *gen));return;}
            choice -= sp->add_cell;
            if (choice < sp->change_cell) {new_anatomy = Anatomy(anatomy.change_random_block(*bp, *gen));return;}
            choice -= sp->change_cell;
            if (choice < sp->remove_cell && anatomy._organism_blocks.size() > sp->min_organism_size) {new_anatomy = Anatomy(anatomy.remove_random_block(*gen));return;}
        } else {
            new_occ = occ.mutate(*occp, *gen);
            new_anatomy = Anatomy(new_occ.compile_code(*occl));

            if (new_anatomy._organism_blocks.empty()) {
                new_anatomy = std::move(Anatomy(anatomy));
                new_occ = OrganismConstructionCode(occ);
            }
            return;
        }
    }
    //if not mutated.
    new_anatomy = std::move(Anatomy(anatomy));
    new_occ = OrganismConstructionCode(occ);
}

void Organism::mutate_brain(Anatomy &new_anatomy, Brain &new_brain,
                            float &_brain_mutation_rate, lehmer64 *gen) {
    // movers without eyes as well.
    if (sp->do_not_mutate_brains_of_plants && (new_anatomy._mover_blocks == 0 || new_anatomy._eye_blocks == 0)) {
        return;
    }

    if (new_anatomy._eye_blocks == 0 && new_anatomy._mover_blocks == 0) {
        new_brain.set_simple_action_table(brain);
    }

    bool mutate_brain;
    _brain_mutation_rate = brain_mutation_rate;

    if (sp->use_brain_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(*gen) <= sp->brain_mutation_rate_delimiter) {
            _brain_mutation_rate += sp->brain_mutation_rate_mutation_step;
            if (_brain_mutation_rate > 1) {_brain_mutation_rate = 1;}
        } else {
            _brain_mutation_rate -= sp->brain_mutation_rate_mutation_step;
            if (_brain_mutation_rate < sp->brain_min_possible_mutation_rate) {
                _brain_mutation_rate = sp->brain_min_possible_mutation_rate;
            }
        }
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(*gen) <= _brain_mutation_rate;
    } else {
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(*gen) <= sp->global_brain_mutation_rate;
    }

    // if mutate brain
    if (mutate_brain) {
        new_brain.set_simple_action_table(brain.mutate(*gen));
    } else {
        // just copy brain from parent
        new_brain.set_simple_action_table(brain);
    }
}

int Organism::mutate_move_range(SimulationParameters *sp, lehmer64 *gen, int parent_move_range) {
    if (sp->set_fixed_move_range) {return parent_move_range;}

    auto child_move_range = parent_move_range;
    if (std::uniform_real_distribution<float>(0, 1)(*gen) <= sp->move_range_delimiter) {
        child_move_range += 1;
        if (child_move_range > sp->max_move_range) {return sp->max_move_range;}
        return child_move_range;
    } else {
        child_move_range -= 1;
        if (child_move_range < sp->min_move_range) {return sp->min_move_range;}
        return child_move_range;
    }
}

int32_t Organism::create_child(lehmer64 *gen, EngineDataContainer &edc) {
    Anatomy new_anatomy;
    Brain new_brain{};
    OrganismConstructionCode new_occ;

    float _anatomy_mutation_rate = 0;
    float _brain_mutation_rate = 0;

    mutate_anatomy(new_anatomy, _anatomy_mutation_rate, gen, new_occ);
    mutate_brain(new_anatomy, new_brain, _brain_mutation_rate, gen);
    auto child_move_range = mutate_move_range(sp, gen, move_range);

    if (new_anatomy._eye_blocks > 0 && new_anatomy._mover_blocks > 0) {
        if (!sp->use_weighted_brain) {
            new_brain.brain_type = BrainTypes::SimpleBrain;
        } else {
            new_brain.brain_type = BrainTypes::WeightedBrain;
        }
    }
    else {new_brain.brain_type = BrainTypes::RandomActions;}

    auto * child_ptr = OrganismsController::get_new_child_organism(edc);
    child_ptr->rotation = rotation;
    child_ptr->anatomy = std::move(new_anatomy);
    child_ptr->brain = new_brain;
    child_ptr->occ = std::move(new_occ);
    child_ptr->sp = sp;
    child_ptr->bp = bp;
    child_ptr->occl = occl;
    child_ptr->occp = occp;
    child_ptr->move_range = child_move_range;
    child_ptr->anatomy_mutation_rate = _anatomy_mutation_rate;
    child_ptr->brain_mutation_rate = _brain_mutation_rate;
    child_ptr->init_values();

    return child_ptr->vector_index;
}

void Organism::think_decision(std::vector<Observation> &organism_observations, lehmer64 *mt) {
    if (anatomy._mover_blocks == 0) { return;}
    if (move_counter == 0) { //if organism can make new move
        auto new_decision = brain.get_decision(organism_observations, rotation, *mt, sp->look_range, sp->threshold_move);
        if (new_decision.decision != BrainDecision::DoNothing) {
            last_decision_observation = new_decision;
        } else {
            if (!sp->no_random_decisions) {
                last_decision_observation = Brain::get_random_action(*mt);
            }
        }
//        if (new_decision.decision != BrainDecision::DoNothing
//            && new_decision.observation.distance > last_decision_observation.observation.distance) {
//            last_decision_observation = new_decision;
//            return;
//        }
//
//        if (new_decision.decision != BrainDecision::DoNothing
//            && last_decision_observation.time > max_decision_lifetime) {
//            last_decision_observation = new_decision;
//            return;
//        }
//
//        if (last_decision_observation.time > max_do_nothing_lifetime) {
//            last_decision_observation = brain.get_random_action(*mt);
//            return;
//        }
//
//        last_decision_observation.time++;
    }
}

void Organism::move_organism(Organism &organism) {
    x = organism.x;
    y = organism.y;
    life_points = organism.life_points;
    damage = organism.damage;
    max_lifetime = organism.max_lifetime;
    lifetime = organism.lifetime;
    anatomy_mutation_rate = organism.anatomy_mutation_rate;
    brain_mutation_rate = organism.brain_mutation_rate;
    food_collected = organism.food_collected;
    food_needed = organism.food_needed;
    multiplier = organism.multiplier;
    move_range = organism.move_range;
    rotation = organism.rotation;
    move_counter = organism.move_counter;
    max_decision_lifetime = organism.max_decision_lifetime;
    max_do_nothing_lifetime = organism.max_do_nothing_lifetime;

    brain = organism.brain;
    anatomy = std::move(organism.anatomy);
    occ = std::move(organism.occ);

    sp = organism.sp;
    bp = organism.bp;
    occp = organism.occp;
    occl = organism.occl;
}

void Organism::kill_organism(EngineDataContainer &edc) {
    if (is_dead) { return;}
    OrganismsController::free_main_organism(this, edc);
}
