// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

#include <utility>

#include "../../SimulationEngine/OrganismsController.h"
#include "AnatomyContainers.h"

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
        multiplier *= anatomy.c["producer"];
    }

    if (anatomy.c["eye"] == 0) { brain.brain_type = BrainTypes::RandomActions;}

    if (sp->use_weighted_brain && brain.brain_type == BrainTypes::SimpleBrain) {
        brain.convert_simple_to_weighted();
    } else if (!sp->use_weighted_brain && brain.brain_type == BrainTypes::WeightedBrain) {
        brain.convert_weighted_to_simple(sp->threshold_move);
    }

    if (sp->use_continuous_movement) {cdata = ContinuousData{float(x), float(y), 0, 0, 0, 0};}
}

float Organism::calculate_max_life() {
    life_points = 0;
    for (auto& item: anatomy.organism_blocks) { life_points += bp->pa[int(item.type) - 1].life_point_amount;}
    return life_points;
}

int Organism::calculate_organism_lifetime() {
    float lifetime_weights = 0;
    for (auto & block: anatomy.organism_blocks) { lifetime_weights += bp->pa[(int)block.type - 1].lifetime_weight;}
    max_lifetime = static_cast<int>(lifetime_weights * sp->lifespan_multiplier);
    return max_lifetime;
}

float Organism::calculate_food_needed() {
    food_needed = sp->extra_reproduction_cost + sp->extra_mover_reproductive_cost * (anatomy.c["mover"] > 0);
    for (auto & block: anatomy.organism_blocks) { food_needed += bp->pa[(int)block.type - 1].food_cost;}
    return food_needed;
}

void Organism::mutate_anatomy(Anatomy &new_anatomy, float &_anatomy_mutation_rate, lehmer64 &gen,
                              OrganismConstructionCode &new_occ) {
    bool mutate_anatomy;
    _anatomy_mutation_rate = anatomy_mutation_rate;

    if (sp->use_anatomy_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(gen) <= sp->anatomy_mutation_rate_delimiter) {
            _anatomy_mutation_rate += sp->anatomy_mutations_rate_mutation_step;
            if (_anatomy_mutation_rate > 1) {_anatomy_mutation_rate = 1;}
        } else {
            _anatomy_mutation_rate -= sp->anatomy_mutations_rate_mutation_step;
            if (_anatomy_mutation_rate < sp->anatomy_min_possible_mutation_rate) {
                _anatomy_mutation_rate = sp->anatomy_min_possible_mutation_rate;
            }
        }
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(gen) <= _anatomy_mutation_rate;
    } else {
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(gen) <= sp->global_anatomy_mutation_rate;
    }

    if (mutate_anatomy) {
        if (!sp->use_occ) {
            int total_chance = 0;
            total_chance += sp->add_cell;
            total_chance += sp->change_cell;
            total_chance += sp->remove_cell;

            int choice = std::uniform_int_distribution<int>(0, total_chance)(gen);

            if (choice < sp->add_cell) {new_anatomy = Anatomy(anatomy.add_random_block(*bp, gen));return;}
            choice -= sp->add_cell;
            if (choice < sp->change_cell) {new_anatomy = Anatomy(anatomy.change_random_block(*bp, gen));return;}
            choice -= sp->change_cell;
            if (choice < sp->remove_cell && anatomy.organism_blocks.size() > sp->min_organism_size) { new_anatomy = Anatomy(anatomy.remove_random_block(gen));return;}
        } else {
            new_occ = occ.mutate(*occp, gen);
            new_anatomy = Anatomy(new_occ.compile_code(*occl));

            if (new_anatomy.organism_blocks.empty()) {
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
                            float &_brain_mutation_rate, lehmer64 &gen) {
    // movers without eyes as well.
    if (sp->do_not_mutate_brains_of_plants && (new_anatomy.c["mover"] == 0 || new_anatomy.c["eye"] == 0)) {
        return;
    }

    bool mutate_brain;
    _brain_mutation_rate = brain_mutation_rate;

    if (sp->use_brain_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(gen) <= sp->brain_mutation_rate_delimiter) {
            _brain_mutation_rate += sp->brain_mutation_rate_mutation_step;
            if (_brain_mutation_rate > 1) {_brain_mutation_rate = 1;}
        } else {
            _brain_mutation_rate -= sp->brain_mutation_rate_mutation_step;
            if (_brain_mutation_rate < sp->brain_min_possible_mutation_rate) {
                _brain_mutation_rate = sp->brain_min_possible_mutation_rate;
            }
        }
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(gen) <= _brain_mutation_rate;
    } else {
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(gen) <= sp->global_brain_mutation_rate;
    }

    // if mutate brain
    if (mutate_brain) {
        new_brain = brain.mutate(gen, *sp);
    } else {
        // just copy brain from parent
        new_brain = Brain{brain};
    }
}

int Organism::mutate_move_range(SimulationParameters *sp, lehmer64 &gen, int parent_move_range) {
    if (sp->set_fixed_move_range) {return parent_move_range;}

    auto child_move_range = parent_move_range;
    if (std::uniform_real_distribution<float>(0, 1)(gen) <= sp->move_range_delimiter) {
        child_move_range += 1;
        if (child_move_range > sp->max_move_range) {return sp->max_move_range;}
        return child_move_range;
    } else {
        child_move_range -= 1;
        if (child_move_range < sp->min_move_range) {return sp->min_move_range;}
        return child_move_range;
    }
}

int32_t Organism::create_child(lehmer64 &gen, EngineDataContainer &edc) {
    Anatomy new_anatomy;
    Brain new_brain{};
    OrganismConstructionCode new_occ;

    float _anatomy_mutation_rate = 0;
    float _brain_mutation_rate = 0;

    mutate_anatomy(new_anatomy, _anatomy_mutation_rate, gen, new_occ);
    mutate_brain(new_anatomy, new_brain, _brain_mutation_rate, gen);
    auto child_move_range = mutate_move_range(sp, gen, move_range);

    if (new_anatomy.c["eye"] > 0 && new_anatomy.c["mover"] > 0) {
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

void Organism::think_decision(std::vector<Observation> &organism_observations, lehmer64 &gen) {
    if (anatomy.c["mover"] == 0) { return;}
    if (move_counter == 0) { //if organism can make new move
        if (sp->use_continuous_movement) {
            calculate_continuous_decision(organism_observations, gen);
        } else {
            calculate_discrete_decision(organism_observations, gen);
        }
    }
}

void Organism::calculate_continuous_decision(std::vector<Observation> &organism_observations, lehmer64 &gen) {
    auto [wdir, ho] = brain.get_global_weighted_direction(organism_observations, sp->look_range, rotation);

    if (brain.brain_type == BrainTypes::RandomActions || (!ho && !sp->no_random_decisions)) {
        wdir[0] = std::uniform_real_distribution<float>(-1., 1.)(gen);
        wdir[1] = std::uniform_real_distribution<float>(-1., 1.)(gen);
        wdir[2] = std::uniform_real_distribution<float>(-1., 1.)(gen);
        wdir[3] = std::uniform_real_distribution<float>(-1., 1.)(gen);
    }

    cdata.p_fx += (wdir[3] - wdir[1]) * 0.5f;
    cdata.p_fy += (wdir[2] - wdir[0]) * 0.5f;
}

void Organism::calculate_discrete_decision(std::vector<Observation> &organism_observations, lehmer64 &gen) {
    auto new_decision = brain.get_decision(organism_observations, rotation, gen, sp->look_range, sp->threshold_move);
    if (new_decision.decision != BrainDecision::DoNothing) {
        last_decision_observation = new_decision;
    } else {
        if (!sp->no_random_decisions) {
            last_decision_observation = Brain::get_random_action(gen);
        }
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

    cdata = organism.cdata;
    brain = organism.brain;
    anatomy = std::move(organism.anatomy);
    occ = std::move(organism.occ);

    sp = organism.sp;
    bp = organism.bp;
    occp = organism.occp;
    occl = organism.occl;
}

void Organism::copy_organism(const Organism &organism) {
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

    cdata = organism.cdata;
    brain = organism.brain;
    anatomy = organism.anatomy;
    occ = organism.occ;

    sp = organism.sp;
    bp = organism.bp;
    occp = organism.occp;
    occl = organism.occl;

    init_values();
}

void Organism::kill_organism(EngineDataContainer &edc) {
    if (is_dead) { return;}
    OrganismsController::free_main_organism(this, edc);
}
