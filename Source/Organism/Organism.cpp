// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

#include <utility>

#include "SimulationEngine/OrganismsController.h"
#include "Anatomy/AnatomyContainers.h"

using BT = BlockTypes;

Organism::Organism(int x, int y, Rotation rotation, Anatomy anatomy, Brain brain, OrganismConstructionCode occ,
                   SimulationParameters *sp, OrganismBlockParameters *block_parameters, OCCParameters *occp,
                   OCCLogicContainer *occl, int move_range, float anatomy_mutation_rate, float brain_mutation_rate,
                   bool no_init) :
        anatomy(anatomy), sp(sp), bp(block_parameters), brain(brain), occ(occ), occp(occp), occl(occl),
        OrganismData(x,
                     y,
                     rotation,
                     move_range,
                     anatomy_mutation_rate,
                     brain_mutation_rate) {

    if (!no_init) {pre_init(); init_values();}
}

Organism::Organism(Organism *organism): anatomy(organism->anatomy), sp(organism->sp), occp(organism->occp), occl(organism->occl),
                                        bp(organism->bp), brain(organism->brain), occ(organism->occ),
                                        OrganismData(organism->x,
                                                     organism->y,
                                                     organism->rotation,
                                                     organism->move_range,
                                                     organism->anatomy_mutation_rate,
                                                     organism->brain_mutation_rate) {
    pre_init();
    init_values();
}

void Organism::pre_init() {
    if (!sp->growth_of_organisms) {
        c = anatomy.c;
        size = -1;
        is_adult = true;
    } else {
        c = make_anatomy_counters();
        for (int i = 0; i < std::min<int64_t>(
                sp->starting_organism_size, anatomy.organism_blocks.size()); i++){
            c[anatomy.organism_blocks[i].type]++;
        }
        size = std::min<int>(sp->starting_organism_size, anatomy.organism_blocks.size());
        is_adult = size == anatomy.organism_blocks.size();
    }

    int max_point = sp->growth_of_organisms ? size : anatomy.organism_blocks.size();
    mass = std::accumulate(anatomy.organism_blocks.begin(), anatomy.organism_blocks.begin()+max_point, 0,
    [&](auto sum, auto & item){return sum + bp->pa[int(item.type)-1].food_cost;});

    lifetime = 0;
    damage = 0;

    cdata = ContinuousData{};
}

void Organism::init_values() {
    calculate_max_life();
    calculate_organism_lifetime();
    calculate_food_needed();

    calc_pos_shift();
    calc_food_multiplier();
    init_brain_type();
    try_convert_brain();
    try_init_cdata();
}

void Organism::try_init_cdata() {
    if (sp->use_continuous_movement && !cdata.initialized) {
        cdata = ContinuousData{float(x), float(y), 0, 0, 0, 0, true};
    } else {
        cdata.initialized = false;
    }
}

void Organism::try_convert_brain() {
    if (sp->use_weighted_brain && brain.brain_type == BrainTypes::SimpleBrain) {
        brain.convert_simple_to_weighted();
    } else if (!sp->use_weighted_brain && brain.brain_type == BrainTypes::WeightedBrain) {
        brain.convert_weighted_to_simple(sp->threshold_move);
    }
}

void Organism::calc_food_multiplier() {
    food_multiplier = 1;
    if (sp->multiply_food_production_prob) { food_multiplier *= c[BT::ProducerBlock];}
}

void Organism::calc_pos_shift() {
    auto vec = anatomy.recenter_blocks(sp->recenter_to_imaginary_pos);

    //just to reuse rotation of positions logic
    auto temp = BaseSerializedContainer{vec.x, vec.y};
    //because organism can be rotated, the shift positions on grid also need to be rotated.
    x += temp.get_pos(rotation).x;
    y += temp.get_pos(rotation).y;
}

void Organism::init_brain_type() {
    if (c[BT::EyeBlock] == 0)   {brain.brain_type = BrainTypes::RandomActions; return;}
    if (c[BT::MoverBlock] <= 0) {brain.brain_type = BrainTypes::RandomActions; return;}

    if (!sp->use_weighted_brain) {
        brain.brain_type = BrainTypes::SimpleBrain;
    } else {
        brain.brain_type = BrainTypes::WeightedBrain;
    }
}

float Organism::calculate_max_life() {
    int max_point = anatomy.organism_blocks.size();
    if (sp->growth_of_organisms) {max_point = size;}
    life_points = std::accumulate(anatomy.organism_blocks.begin(), anatomy.organism_blocks.begin()+max_point,0,
    [&](auto sum, auto & item){return sum + bp->pa[int(item.type)-1].life_point_amount;});

    return life_points;
}

int Organism::calculate_organism_lifetime() {
    int max_point = anatomy.organism_blocks.size();
    if (sp->growth_of_organisms) {max_point = size;}
    float lifetime_weights = std::accumulate(anatomy.organism_blocks.begin(), anatomy.organism_blocks.begin()+max_point, 0,
       [&](auto sum, auto & item){return sum + bp->pa[int(item.type)-1].lifetime_weight;});

    max_lifetime = static_cast<int>(lifetime_weights * sp->lifespan_multiplier);
    return max_lifetime;
}

float Organism::calculate_food_needed() {
    food_needed = sp->extra_reproduction_cost + sp->extra_mover_reproductive_cost * (anatomy.c[BT::MoverBlock] > 0);

    int max_point = anatomy.organism_blocks.size();
    if (sp->growth_of_organisms) {max_point = size;}
    food_needed = std::accumulate(anatomy.organism_blocks.begin(), anatomy.organism_blocks.begin()+max_point, food_needed,
    [&](auto sum, auto & item){return sum + bp->pa[int(item.type)-1].food_cost;});

    return food_needed;
}

void Organism::mutate_anatomy(Anatomy &new_anatomy, float &_anatomy_mutation_rate, lehmer64 &gen,
                              OrganismConstructionCode &new_occ) {
    bool mutate_anatomy;
    _anatomy_mutation_rate = anatomy_mutation_rate;

    if (sp->use_anatomy_evolved_mutation_rate) {
        mutate_mutation_rate(_anatomy_mutation_rate, gen);
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(gen) <= _anatomy_mutation_rate;
    } else {
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(gen) <= sp->global_anatomy_mutation_rate;
    }

    if (!mutate_anatomy) {
        new_anatomy = std::move(Anatomy(anatomy));
        new_occ = OrganismConstructionCode(occ);
        return;
    }

    if (sp->use_occ) {
        mutate_occ(new_anatomy, gen, new_occ);return;
    } else {
        mutate_legacy(new_anatomy, gen);
#ifdef __DEBUG__
        if(new_anatomy.organism_blocks.empty()) {
           throw std::logic_error("how");
        }
#endif
        return;
    }
}

void Organism::mutate_mutation_rate(float &_anatomy_mutation_rate, lehmer64 &gen) const {
    if (std::uniform_real_distribution<float>(0, 1)(gen) <= sp->anatomy_mutation_rate_delimiter) {
        _anatomy_mutation_rate += sp->anatomy_mutations_rate_mutation_step;
        if (_anatomy_mutation_rate > 1) {_anatomy_mutation_rate = 1;}
    } else {
        _anatomy_mutation_rate -= sp->anatomy_mutations_rate_mutation_step;
        if (_anatomy_mutation_rate < sp->anatomy_min_possible_mutation_rate) {
            _anatomy_mutation_rate = sp->anatomy_min_possible_mutation_rate;
        }
    }
}

void Organism::mutate_occ(Anatomy &new_anatomy, lehmer64 &gen, OrganismConstructionCode &new_occ) {
    new_occ = occ.mutate(*occp, gen);
    new_anatomy = Anatomy(new_occ.compile_code(*occl, sp->growth_of_organisms));

    if (new_anatomy.organism_blocks.empty()) {
        new_anatomy = std::move(Anatomy(anatomy));
        new_occ = OrganismConstructionCode(occ);
    }
}

void Organism::mutate_legacy(Anatomy &new_anatomy, lehmer64 &gen) {
    int total_chance = 0;
    total_chance += sp->add_cell + sp->change_cell + sp->remove_cell;

    int choice = std::uniform_int_distribution<int>(0, total_chance)(gen);

    if (choice < sp->add_cell) { new_anatomy = Anatomy(anatomy.add_random_block(*bp, gen));return;}
    choice -= sp->add_cell;
    if (choice < sp->change_cell) { new_anatomy = Anatomy(anatomy.change_random_block(*bp, gen));return;}
    choice -= sp->change_cell;
    if ((anatomy.organism_blocks.size() > sp->min_organism_size) && choice < sp->remove_cell) {new_anatomy = Anatomy(anatomy.remove_random_block(gen));return;}
    new_anatomy = Anatomy(anatomy);
}

void Organism::mutate_brain(const Anatomy &new_anatomy, Brain &new_brain,
                            float &_brain_mutation_rate, lehmer64 &gen) {
    // movers without eyes as well.
    if (sp->do_not_mutate_brains_of_plants && (new_anatomy.c[BT::MoverBlock] == 0 || new_anatomy.c[BT::EyeBlock] == 0)) {
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
    } else {
        child_move_range -= 1;
        if (child_move_range < sp->min_move_range) {return sp->min_move_range;}
    }
    return child_move_range;
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
    child_ptr->pre_init();
    child_ptr->init_values();

    return child_ptr->vector_index;
}

void Organism::think_decision(std::vector<Observation> &organism_observations, lehmer64 &gen) {
    if (anatomy.c[BT::MoverBlock] == 0) { return;}
    if (move_counter != 0) { return;}

    if (sp->use_continuous_movement) {
        calculate_continuous_decision(organism_observations, gen);
    } else {
        calculate_discrete_decision(organism_observations, gen);
    }
}

void Organism::calculate_continuous_decision(std::vector<Observation> &organism_observations, lehmer64 &gen) {
    auto [wdir, had_observation] = brain.get_global_weighted_direction(organism_observations, sp->look_range, rotation);

    if (brain.brain_type == BrainTypes::RandomActions || (!had_observation && !sp->no_random_decisions)) {
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
        last_decision_observation = new_decision; return;
    }
    if (!sp->no_random_decisions) {
        last_decision_observation = Brain::get_random_action(gen);
    }
}

void Organism::move_organism(Organism &organism) {
    *(OrganismData*)this = *(OrganismData*)&organism;

    brain = organism.brain;
    anatomy = std::move(organism.anatomy);
    occ = std::move(organism.occ);
    c = organism.c;

    sp = organism.sp;
    bp = organism.bp;
    occp = organism.occp;
    occl = organism.occl;
}

void Organism::copy_organism(const Organism &organism) {
    *(OrganismData*)this = *(OrganismData*)&organism;

    brain = organism.brain;
    anatomy = organism.anatomy;
    occ = organism.occ;
    c = organism.c;

    sp = organism.sp;
    bp = organism.bp;
    occp = organism.occp;
    occl = organism.occl;

    pre_init();
    init_values();
}

void Organism::kill_organism(EngineDataContainer &edc) {
    if (is_dead) { return;}
    OrganismsController::free_main_organism(this, edc);
}
