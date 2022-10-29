// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include <random>
#include <exception>

#include "Anatomy.h"
#include "Brain.h"
#include "OrganismConstructionCode.h"
#include "../../Containers/CPU/SimulationParameters.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "Rotation.h"
#include "../../PRNGS/lehmer64.h"
#include "ObservationStuff.h"
#include "Rotation.h"
#include "ObservationStuff.h"

struct EngineDataContainer;

struct OrganismData {
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

    float multiplier = 1;

    int move_range = 1;
    Rotation rotation = Rotation::UP;

    int move_counter = 0;

    //TODO make evolvable
    int max_decision_lifetime = 2;
    int max_do_nothing_lifetime = 3;

    DecisionObservation last_decision_observation = DecisionObservation{};
    BrainDecision last_decision = BrainDecision::MoveUp;

    OrganismData()=default;
    OrganismData(int x, int y, Rotation rotation, int move_range, float anatomy_mutation_rate,
                 float brain_mutation_rate): x(x), y(y), rotation(rotation), anatomy_mutation_rate(anatomy_mutation_rate),
                                             brain_mutation_rate(brain_mutation_rate), move_range(move_range) {};
};

class Organism: public OrganismData {
public:
    Anatomy anatomy;
    Brain brain;
    OrganismConstructionCode occ;
    SimulationParameters* sp = nullptr;
    OrganismBlockParameters* bp = nullptr;
    OCCParameters * occp = nullptr;
    OCCLogicContainer * occl = nullptr;
    int32_t child_pattern_index = -1;
    int32_t vector_index = -1;

    bool is_dead = false;

    float calculate_max_life();
    int calculate_organism_lifetime();
    float calculate_food_needed();

    void mutate_anatomy(Anatomy &new_anatomy, float &_anatomy_mutation_rate, lehmer64 *gen,
                        OrganismConstructionCode &new_occ);
    void mutate_brain(Anatomy &new_anatomy, Brain &new_brain, float &_brain_mutation_rate, lehmer64 *gen);
    static int mutate_move_range(SimulationParameters *sp, lehmer64 *gen, int parent_move_range);

    void think_decision(std::vector<Observation> &organism_observations, lehmer64 *mt);

    void init_values();

    void kill_organism(EngineDataContainer & edc);

    void move_organism(Organism & organism);
    void copy_organism(const Organism & organism);

    Organism & operator=(const Organism & organism)=default;
    Organism()=default;
    Organism(Organism&&)=default;
    Organism(int x, int y, Rotation rotation, Anatomy anatomy, Brain brain, OrganismConstructionCode occ,
             SimulationParameters *sp, OrganismBlockParameters *block_parameters, OCCParameters *occp,
             OCCLogicContainer *occl, int move_range, float anatomy_mutation_rate = 0.05,
             float brain_mutation_rate = 0.1);
    Organism(Organism *organism);
    int32_t create_child(lehmer64 *gen, EngineDataContainer &edc);
};


#endif //THELIFEENGINECPP_ORGANISM_H
