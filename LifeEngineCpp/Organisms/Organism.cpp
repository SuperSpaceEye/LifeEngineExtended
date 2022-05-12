//
// Created by spaceeye on 20.03.2022.
//

#include "Organism.h"

Organism::Organism(int x, int y, bool * can_rotate, Rotation rotation, Anatomy* anatomy,
                   SimulationParameters* sim_parameters, OrganismBlockParameters* block_parameters, std::mt19937* mt):
        x(x), y(y), can_rotate(can_rotate), rotation(rotation), organism_anatomy(anatomy), sim_parameters(sim_parameters),
        block_parameters(block_parameters), mt(mt) {
    calculate_max_life(organism_anatomy);
    calculate_organism_lifetime(organism_anatomy);
}

Organism::~Organism() {
    //delete organism_anatomy;
}

float Organism::calculate_max_life(Anatomy *anatomy) {
    life_points = 0;
    for (auto& item: organism_anatomy->_organism_blocks) {
        switch (item.organism_block.type) {
            case MouthBlock:
                life_points += block_parameters->MouthBlock.life_point_amount;
                break;
            case ProducerBlock:
                life_points += block_parameters->ProducerBlock.life_point_amount;
                break;
            case MoverBlock:
                life_points += block_parameters->MoverBlock.life_point_amount;
                break;
            case KillerBlock:
                life_points += block_parameters->KillerBlock.life_point_amount;
                break;
            case ArmorBlock:
                life_points += block_parameters->ArmorBlock.life_point_amount;
                break;
            case EyeBlock:
                life_points += block_parameters->EyeBlock.life_point_amount;
                break;
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
int Organism::calculate_organism_lifetime(Anatomy *anatomy) {
    max_lifetime = anatomy->_organism_blocks.size() * sim_parameters->lifespan_multiplier;
    return max_lifetime;
}

Organism * Organism::create_child() {
    int total_chance = 0;
    total_chance += sim_parameters->add_cell;
    total_chance += sim_parameters->change_cell;
    total_chance += sim_parameters->remove_cell;
    total_chance += sim_parameters->do_nothing;

    int choice = std::uniform_int_distribution<int>(0, total_chance)(*mt);

    Anatomy* new_anatomy;
    //TODO this is stupid, need refactoring
    if (choice < sim_parameters->add_cell)    {new_anatomy = new Anatomy(organism_anatomy->add_random_block(*block_parameters, *mt));}
    else {choice -= sim_parameters->add_cell;
    if (choice < sim_parameters->change_cell) {new_anatomy = new Anatomy(organism_anatomy->change_random_block(*block_parameters, *mt));}
    else {choice -= sim_parameters->change_cell;
    if (choice < sim_parameters->remove_cell) {new_anatomy = new Anatomy(organism_anatomy->remove_random_block(*mt));}
    else                                     {new_anatomy = new Anatomy(organism_anatomy);
    }}}

    return new Organism(0, 0, can_rotate, rotation, new_anatomy, sim_parameters, block_parameters, mt);
}
