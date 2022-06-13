//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include <random>
#include <exception>

#include "Anatomy.h"
#include "Brain.h"
#include "../../Containers/CPU/SimulationParameters.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "Rotation.h"
#include "../../PRNGS/lehmer64.h"
#include "ObservationStuff.h"

class Organism {
//private:
public:
    //coordinates of a central block of an organism
    int x = 0;
    int y = 0;
    //how much damage organism can sustain.
    float life_points = 0;
    //how much damage organism sustained. If damage > life_points, organism dies
    float damage = 0;

    //an amount of simulation ticks organism can live
    int max_lifetime = 0;
    //how much organism already lived.
    int lifetime = 0;

    float anatomy_mutation_rate = 0.05;
    float brain_mutation_rate = 0.1;

    float food_collected = 0;
    float food_needed = 0;

    int move_range = 1;
    bool * can_rotate = nullptr;
    Rotation rotation = Rotation::UP;

    bool child_ready = false;

    int move_counter = 0;

    //TODO make evolvable
    int max_decision_lifetime = 2;
    int max_do_nothing_lifetime = 4;

    DecisionObservation last_decision = DecisionObservation{};

    std::shared_ptr<Anatomy> organism_anatomy = nullptr;
    std::shared_ptr<Brain> brain = nullptr;
    SimulationParameters* sp = nullptr;
    OrganismBlockParameters* bp = nullptr;
    Organism * child_pattern = nullptr;

    float calculate_max_life();
    int calculate_organism_lifetime();
    float calculate_food_needed();

    void mutate_anatomy(std::shared_ptr<Anatomy> &new_anatomy, float &_anatomy_mutation_rate, lehmer64 *gen);
    void mutate_brain(std::shared_ptr<Anatomy> &new_anatomy, std::shared_ptr<Brain> &new_brain, float &_brain_mutation_rate, lehmer64 *gen);
    static int mutate_move_range(SimulationParameters *sp, lehmer64 *gen, int parent_move_range);

    void think_decision(std::vector<Observation> &organism_observations, lehmer64 *mt);
    //public:
    Organism(int x, int y, bool *can_rotate, Rotation rotation, std::shared_ptr<Anatomy> anatomy,
             std::shared_ptr<Brain> brain, SimulationParameters *sp,
             OrganismBlockParameters *block_parameters, int move_range,
             float anatomy_mutation_rate= 0.05, float brain_mutation_rate= 0.1);
    Organism(Organism *organism);
    Organism()=default;
    ~Organism();
    Organism * create_child(lehmer64 *gen);
};


#endif //THELIFEENGINECPP_ORGANISM_H
