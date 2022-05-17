//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include <random>
#include "Anatomy.h"
#include "Brain.h"
#include "../SimulationParameters.h"
#include "../OrganismBlockParameters.h"
#include "Rotation.h"

//TODO It's stupid, but it is prototype
class Organism {
//private:
public:
    //coordinates of a central block of a cell
    int x = 0;
    int y = 0;
    // how much organism can sustain.
    float life_points = 0;
    // how much damage organism sustained. If damage > life_points, organism dies
    float damage = 0;

    // an amount of ticks organism can live
    int max_lifetime = 0;
    // for how much organism already lived.
    int lifetime = 0;

    float anatomy_mutation_rate = 0.05;
    float brain_mutation_rate = 0.1;

    float food_collected = 0;
    float food_needed = 0;
    //TODO implement rotation
    bool * can_rotate = nullptr;
    Rotation rotation = Rotation::UP;

    bool child_ready = false;

    std::shared_ptr<Anatomy> organism_anatomy = nullptr;
    std::shared_ptr<Brain> brain = nullptr;
    SimulationParameters* sp = nullptr;
    OrganismBlockParameters* bp = nullptr;
    Organism * child_pattern = nullptr;

    std::mt19937* mt = nullptr;

    float calculate_max_life();
    int calculate_organism_lifetime();
    float calculate_food_needed();

    void mutate_anatomy(std::shared_ptr<Anatomy> &new_anatomy, float &_anatomy_mutation_rate);
    void mutate_brain(std::shared_ptr<Anatomy> &new_anatomy, std::shared_ptr<Brain> &new_brain, float &_brain_mutation_rate);
//public:
    Organism(int x, int y, bool *can_rotate, Rotation rotation, std::shared_ptr<Anatomy> anatomy,
             std::shared_ptr<Brain> brain, SimulationParameters *sp,
             OrganismBlockParameters *block_parameters, std::mt19937 *mt, float anatomy_mutation_rate=0.5,
             float brain_mutation_rate=0.5);
    Organism(Organism *organism);
    Organism()=default;
    ~Organism();
    Organism * create_child();
};


#endif //THELIFEENGINECPP_ORGANISM_H
