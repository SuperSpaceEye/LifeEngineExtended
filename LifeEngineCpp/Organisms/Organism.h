//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include <random>
#include "Anatomy.h"
#include "../SimulationParameters.h"
#include "../OrganismBlockParameters.h"

enum class Rotation {
    UP,
    RIGHT,
    DOWN,
    LEFT
};

struct Coordinates {
    int x;
    int y;
};

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

    float food_collected = 0;
    //TODO implement rotation
    bool * can_rotate = nullptr;
    Rotation rotation = Rotation::UP;

    Anatomy * organism_anatomy = nullptr;
    SimulationParameters* sim_parameters = nullptr;
    OrganismBlockParameters* block_parameters = nullptr;

    std::mt19937* mt;

    float calculate_max_life(Anatomy *anatomy);
    int calculate_organism_lifetime(Anatomy *anatomy);
//public:
    Organism(int x, int y, bool* can_rotate, Rotation rotation, Anatomy * anatomy,
             SimulationParameters* sim_parameters, OrganismBlockParameters* block_parameters, std::mt19937* mt);
    Organism()=default;
    ~Organism();
    Organism * create_child();
};


#endif //THELIFEENGINECPP_ORGANISM_H
