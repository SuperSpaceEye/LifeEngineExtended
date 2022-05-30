//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

#include <utility>
#include "Rotation.h"

Organism::Organism(int x, int y, bool *can_rotate, Rotation rotation, std::shared_ptr<Anatomy> anatomy,
                   std::shared_ptr<Brain> brain, SimulationParameters *sp,
                   OrganismBlockParameters *block_parameters, int move_range, float anatomy_mutation_rate,
                   float brain_mutation_rate) :
//        x(x), y(y), can_rotate(can_rotate), rotation(rotation), organism_anatomy(std::move(anatomy)), sp(sp),
//        bp(block_parameters), mt(mt), brain(std::move(brain)) {
        x(x), y(y), can_rotate(can_rotate), rotation(rotation), organism_anatomy(anatomy), sp(sp),
        bp(block_parameters), brain(brain), anatomy_mutation_rate(anatomy_mutation_rate),
        brain_mutation_rate(brain_mutation_rate), move_range(move_range) {
    calculate_max_life();
    calculate_organism_lifetime();
    calculate_food_needed();
}

Organism::Organism(Organism *organism): x(organism->x), y(organism->y), can_rotate(organism->can_rotate),
                                        rotation(organism->rotation), organism_anatomy(organism->organism_anatomy), sp(organism->sp),
                                        bp(organism->bp), brain(organism->brain),
                                        anatomy_mutation_rate(organism->anatomy_mutation_rate),
                                        brain_mutation_rate(organism->brain_mutation_rate),
                                        move_range(organism->move_range){
    calculate_max_life();
    calculate_organism_lifetime();
    calculate_food_needed();
}

Organism::~Organism() {
    //delete organism_anatomy;
    delete child_pattern;
}

float Organism::calculate_max_life() {
    life_points = 0;
    for (auto& item: organism_anatomy->_organism_blocks) {
        switch (item.type) {
            case MouthBlock:    life_points += bp->MouthBlock.   life_point_amount; break;
            case ProducerBlock: life_points += bp->ProducerBlock.life_point_amount; break;
            case MoverBlock:    life_points += bp->MoverBlock.   life_point_amount; break;
            case KillerBlock:   life_points += bp->KillerBlock.  life_point_amount; break;
            case ArmorBlock:    life_points += bp->ArmorBlock.   life_point_amount; break;
            case EyeBlock:      life_points += bp->EyeBlock.     life_point_amount; break;
            default: throw std::runtime_error("Unknown block");
        }
    }
    return life_points;
}

//TODO for the future
int Organism::calculate_organism_lifetime() {
    max_lifetime = organism_anatomy->_organism_blocks.size() * sp->lifespan_multiplier;
    return max_lifetime;
}

float Organism::calculate_food_needed() {
    food_needed = sp->extra_reproduction_cost;
    for (auto & block: organism_anatomy->_organism_blocks) {
        switch (block.type) {
            case MouthBlock:    food_needed += bp->MouthBlock.   food_cost_modifier; break;
            case ProducerBlock: food_needed += bp->ProducerBlock.food_cost_modifier; break;
            case MoverBlock:    food_needed += bp->MoverBlock.   food_cost_modifier; break;
            case KillerBlock:   food_needed += bp->KillerBlock.  food_cost_modifier; break;
            case ArmorBlock:    food_needed += bp->ArmorBlock.   food_cost_modifier; break;
            case EyeBlock:      food_needed += bp->EyeBlock.     food_cost_modifier; break;
            default: throw std::runtime_error("Unknown block");
        }
    }
    return food_needed;
}

void Organism::mutate_anatomy(std::shared_ptr<Anatomy> &new_anatomy, float &_anatomy_mutation_rate, boost::mt19937 *mt) {
    bool mutate_anatomy;
    _anatomy_mutation_rate = anatomy_mutation_rate;

    if (sp->use_anatomy_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(*mt) <= sp->anatomy_mutation_rate_delimiter) {
            _anatomy_mutation_rate += sp->anatomy_mutations_rate_mutation_modifier;
        } else {
            _anatomy_mutation_rate -= sp->anatomy_mutations_rate_mutation_modifier;
            if (_anatomy_mutation_rate < sp->anatomy_min_possible_mutation_rate) {
                _anatomy_mutation_rate = sp->anatomy_min_possible_mutation_rate;
            }
        }
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(*mt) <= _anatomy_mutation_rate;
    } else {
        mutate_anatomy = std::uniform_real_distribution<float>(0, 1)(*mt) <= sp->global_anatomy_mutation_rate;
    }

    if (mutate_anatomy) {
        int total_chance = 0;
        total_chance += sp->add_cell;
        total_chance += sp->change_cell;
        total_chance += sp->remove_cell;

        int choice = std::uniform_int_distribution<int>(0, total_chance)(*mt);

        if (choice < sp->add_cell) {new_anatomy.reset(new Anatomy(organism_anatomy->add_random_block(*bp, *mt)));return;}
        choice -= sp->add_cell;
        if (choice < sp->change_cell) {new_anatomy.reset(new Anatomy(organism_anatomy->change_random_block(*bp, *mt)));return;}
        choice -= sp->change_cell;
        if (choice < sp->remove_cell && organism_anatomy->_organism_blocks.size() > sp->min_organism_size) {new_anatomy.reset(new Anatomy(organism_anatomy->remove_random_block(*mt)));return;}
    }
    //if not mutated.
    new_anatomy.reset(new Anatomy(organism_anatomy));
}

void Organism::mutate_brain(std::shared_ptr<Anatomy> &new_anatomy, std::shared_ptr<Brain> &new_brain,
                            float &_brain_mutation_rate, boost::mt19937 *mt) {
    bool mutate_brain;
    _brain_mutation_rate = brain_mutation_rate;

    if (sp->use_brain_evolved_mutation_rate) {
        if (std::uniform_real_distribution<float>(0,1)(*mt) <= sp->brain_mutation_rate_delimiter) {
            _brain_mutation_rate += sp->brain_mutation_rate_mutation_modifier;
        } else {
            _brain_mutation_rate -= sp->brain_mutation_rate_mutation_modifier;
            if (_brain_mutation_rate < sp->brain_min_possible_mutation_rate) {
                _brain_mutation_rate = sp->brain_min_possible_mutation_rate;
            }
        }
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(*mt) <= _brain_mutation_rate;
    } else {
        mutate_brain = std::uniform_real_distribution<float>(0, 1)(*mt) <= sp->global_brain_mutation_rate;
    }

    // if mutate brain
    if (mutate_brain && new_anatomy->_eye_blocks > 0 && new_anatomy->_mover_blocks > 0) {
        new_brain.reset(brain->mutate(*mt));
    } else {
        // just copy brain from parent
        new_brain.reset(new Brain(brain));
    }
}

int Organism::mutate_move_range(SimulationParameters *sp, boost::mt19937 *mt, int parent_move_range) {
    if (sp->set_fixed_move_range) {return parent_move_range;}

    auto child_move_range = parent_move_range;
    if (std::uniform_real_distribution<float>(0, 1)(*mt) <= sp->move_range_delimiter) {
        child_move_range += 1;
        if (child_move_range > sp->max_move_range) {return sp->max_move_range;}
        return child_move_range;
    } else {
        child_move_range -= 1;
        if (child_move_range < sp->min_move_range) {return sp->min_move_range;}
        return child_move_range;
    }
}

Organism * Organism::create_child(boost::mt19937 *mt) {
    std::shared_ptr<Anatomy> new_anatomy;
    std::shared_ptr<Brain>   new_brain;

    float _anatomy_mutation_rate = 0;
    float _brain_mutation_rate = 0;

    mutate_anatomy(new_anatomy, _anatomy_mutation_rate, mt);
    mutate_brain(new_anatomy, new_brain, _brain_mutation_rate, mt);
    auto child_move_range = mutate_move_range(sp, mt, move_range);

    if (new_anatomy->_eye_blocks > 0 && new_anatomy->_mover_blocks > 0) {new_brain->brain_type = BrainTypes::SimpleBrain;}
    else {new_brain->brain_type = BrainTypes::RandomActions;}

    return new Organism(0,
                        0,
                        can_rotate,
                        rotation,
                        new_anatomy,
                        new_brain,
                        sp,
                        bp,
                        child_move_range,
                        _anatomy_mutation_rate,
                        _brain_mutation_rate);
}