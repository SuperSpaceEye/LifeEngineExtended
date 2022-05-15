//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

#include <utility>
#include "Rotation.h"

Organism::Organism(int x, int y, bool * can_rotate, Rotation rotation, std::shared_ptr<Anatomy> anatomy,
                   SimulationParameters* sp, OrganismBlockParameters* block_parameters, std::mt19937* mt):
        x(x), y(y), can_rotate(can_rotate), rotation(rotation), organism_anatomy(std::move(anatomy)), sp(sp),
        bp(block_parameters), mt(mt) {
    calculate_max_life(organism_anatomy);
    calculate_organism_lifetime(organism_anatomy);
    calculate_food_needed(organism_anatomy);
    brain = new Brain{mt, BrainTypes::RandomActions};
}

Organism::Organism(Organism *organism): x(organism->x), y(organism->y), can_rotate(organism->can_rotate),
                                        rotation(organism->rotation), organism_anatomy(organism->organism_anatomy), sp(organism->sp),
                                        bp(organism->bp), mt(organism->mt), brain(organism->brain) {
    calculate_max_life(organism_anatomy);
    calculate_organism_lifetime(organism_anatomy);
    calculate_food_needed(organism_anatomy);
}

Organism::~Organism() {
    //delete organism_anatomy;
    delete child_pattern;
}

float Organism::calculate_max_life(const std::shared_ptr<Anatomy>& anatomy) {
    life_points = 0;
    for (auto& item: anatomy->_organism_blocks) {
        switch (item.organism_block.type) {
            case MouthBlock:    life_points += bp->MouthBlock.   life_point_amount; break;
            case ProducerBlock: life_points += bp->ProducerBlock.life_point_amount; break;
            case MoverBlock:    life_points += bp->MoverBlock.   life_point_amount; break;
            case KillerBlock:   life_points += bp->KillerBlock.  life_point_amount; break;
            case ArmorBlock:    life_points += bp->ArmorBlock.   life_point_amount; break;
            case EyeBlock:      life_points += bp->EyeBlock.     life_point_amount; break;
            //These cases can't happen
            case EmptyBlock:
            case FoodBlock:
            case WallBlock:
                break;
        }
    }
    return life_points;
}

//TODO for the future
int Organism::calculate_organism_lifetime(const std::shared_ptr<Anatomy>& anatomy) {
    max_lifetime = anatomy->_organism_blocks.size() * sp->lifespan_multiplier;
    return max_lifetime;
}

float Organism::calculate_food_needed(const std::shared_ptr<Anatomy>& anatomy) {
    food_needed = 0;
    for (auto & block: anatomy->_organism_blocks) {
        switch (block.organism_block.type) {
            case MouthBlock:    food_needed += bp->MouthBlock.   food_cost_modifier; break;
            case ProducerBlock: food_needed += bp->ProducerBlock.food_cost_modifier; break;
            case MoverBlock:    food_needed += bp->MoverBlock.   food_cost_modifier; break;
            case KillerBlock:   food_needed += bp->KillerBlock.  food_cost_modifier; break;
            case ArmorBlock:    food_needed += bp->ArmorBlock.   food_cost_modifier; break;
            case EyeBlock:      food_needed += bp->EyeBlock.     food_cost_modifier; break;
            case EmptyBlock:
            case FoodBlock:
            case WallBlock:
                break;
        }
    }
    return food_needed;
}

Organism * Organism::create_child() {
    bool mutate;
    std::shared_ptr<Anatomy> new_anatomy;

    if (sp->use_evolved_mutation_rate) {

    } else {
        mutate = std::uniform_real_distribution<float>(0, 1)(*mt) <= sp->global_mutation_rate;
    }

    if (mutate) {
        int total_chance = 0;
        total_chance += sp->add_cell;
        total_chance += sp->change_cell;
        total_chance += sp->remove_cell;

        int choice = std::uniform_int_distribution<int>(0, total_chance)(*mt);

        //TODO i don't like this stairs of if's end else's.
        if (choice < sp->add_cell) {
            new_anatomy.reset(new Anatomy(organism_anatomy->add_random_block(*bp, *mt)));
        } else {
            choice -= sp->add_cell;
            if (choice < sp->change_cell) {
                new_anatomy.reset(new Anatomy(organism_anatomy->change_random_block(*bp, *mt)));
            } else {
                choice -= sp->change_cell;
                if (choice < sp->remove_cell && organism_anatomy->_organism_blocks.size() > 1) {
                    new_anatomy.reset(new Anatomy(organism_anatomy->remove_random_block(*mt)));
                }
            }
        }
    } else {
        new_anatomy.reset(new Anatomy(organism_anatomy));
    }

    //TODO VERY IMPORTANT! leak here? when try_make_child
    return new Organism(0, 0, can_rotate, rotation, new_anatomy, sp, bp, mt);
}
