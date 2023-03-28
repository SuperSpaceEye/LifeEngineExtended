// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include <random>
#include <exception>

#include "Stuff/external/ArrayView.h"

#include "Brain/Brain.h"
#include "Brain/Observation.h"
#include "Stuff/enums/Rotation.h"
#include "Containers/SimulationParameters.h"
#include "Containers/OrganismBlockParameters.h"
#include "OCC/OrganismConstructionCode.h"
#include "PRNGS/lehmer64.h"
#include "Anatomy/Anatomy.h"

struct EngineDataContainer;

struct ContinuousData {
    // physics
    float p_x = 0;
    float p_y = 0;
    //velocity
    float p_vx = 0;
    float p_vy = 0;
    //force
    float p_fx = 0;
    float p_fy = 0;
    bool initialized = false;
};

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
    float mass = 0;

    float multiplier = 1;

    int move_range = 1;
    Rotation rotation = Rotation::UP;

    int move_counter = 0;

    uint32_t size = -1;

    bool is_adult = false;

    ContinuousData cdata;

    DecisionObservation last_decision_observation = DecisionObservation{};
    BrainDecision last_decision = BrainDecision::MoveUp;

    AnatomyCounters<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES> c = make_anatomy_counters();

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

    void mutate_anatomy(Anatomy &new_anatomy, float &_anatomy_mutation_rate, lehmer64 &gen,
                        OrganismConstructionCode &new_occ);
    void mutate_brain(const Anatomy &new_anatomy, Brain &new_brain, float &_brain_mutation_rate, lehmer64 &gen);
    static int mutate_move_range(SimulationParameters *sp, lehmer64 &gen, int parent_move_range);

    void think_decision(std::vector<Observation> &organism_observations, lehmer64 &gen);

    void init_values();

    void kill_organism(EngineDataContainer & edc);

    void move_organism(Organism & organism);
    void copy_organism(const Organism & organism);

    Organism & operator=(const Organism & organism)=default;
    Organism()=default;
    Organism(Organism&&)=default;
    Organism(int x, int y, Rotation rotation, Anatomy anatomy, Brain brain, OrganismConstructionCode occ,
             SimulationParameters *sp, OrganismBlockParameters *block_parameters, OCCParameters *occp,
             OCCLogicContainer *occl, int move_range, float anatomy_mutation_rate, float brain_mutation_rate,
             bool no_init=false);
    explicit Organism(Organism *organism);
    int32_t create_child(lehmer64 &gen, EngineDataContainer &edc);

    void calculate_discrete_decision(std::vector<Observation> &organism_observations, lehmer64 &gen);

    void calculate_continuous_decision(std::vector<Observation> &organism_observations, lehmer64 &gen);

    //TODO make init_values include pre_init, but rename original init_values to something else to separate usages
    void pre_init();
    
//    using BlockTypes   = BlockTypes;
//    using SerializedAdjacentSpaceContainer = SerializedAdjacentSpaceContainer;
//    using SerializedOrganismBlockContainer = SerializedOrganismBlockContainer;

//    array_view1d<SerializedOrganismBlockContainer> organism_blocks_view;
//    array_view1d<SerializedOrganismBlockContainer> eye_blocks_view;
//    array_view1d<std::vector<SerializedAdjacentSpaceContainer>> producing_space_view;
//
//    array_view1d<SerializedAdjacentSpaceContainer> eating_view;
//    array_view1d<SerializedAdjacentSpaceContainer> killing_view;

//    void make_views() {
////        organism_blocks_view = array_view1d<SerializedOrganismBlockContainer>{(SerializedOrganismBlockContainer*)anatomy.organism_blocks.data(), std::min<uint64_t>(size, anatomy.organism_blocks.size())};
////        if (!anatomy.eye_block_vec.empty())   {eye_blocks_view      = array_view1d<SerializedOrganismBlockContainer>{(SerializedOrganismBlockContainer*)anatomy.eye_block_vec.data(),   std::min<uint64_t>(c[BlockTypes::EyeBlock], anatomy.eye_block_vec.size())};}
////        if (!anatomy.producing_space.empty()) {producing_space_view = array_view1d<std::vector<SerializedAdjacentSpaceContainer>>{(std::vector<SerializedAdjacentSpaceContainer>*)anatomy.producing_space.data(), std::min<uint64_t>(c[BlockTypes::ProducerBlock], anatomy.producing_space.size())};}
////
////        if (!anatomy.eating_space.empty())  {eating_view   = array_view1d<SerializedAdjacentSpaceContainer>{(SerializedAdjacentSpaceContainer*)anatomy.eating_space.data(),   std::min<size_t>(anatomy.eating_mask[c[BlockTypes::MouthBlock]-1], anatomy.eating_space.size())};}
////        if (!anatomy.killing_space.empty()) {killing_view  = array_view1d<SerializedAdjacentSpaceContainer>{(SerializedAdjacentSpaceContainer*)anatomy.killing_space.data(),  std::min<size_t>(anatomy.killer_mask[c[BlockTypes::KillerBlock]-1], anatomy.killing_space.size())};}
//    }

    using BT = BlockTypes;
    using SOBC = SerializedOrganismBlockContainer;
    using SASC = SerializedAdjacentSpaceContainer;

    inline auto get_organism_blocks_view() {return array_view1d<SOBC>{(SOBC*)anatomy.organism_blocks.data(), std::min<uint32_t>(size, anatomy.organism_blocks.size())};}
    inline auto get_eye_block_vec_view  () {return array_view1d<SOBC>{(SOBC*)anatomy.eye_block_vec.data(),   std::min<uint32_t>(c[BT::EyeBlock], anatomy.eye_block_vec.size())};}
    inline auto get_producing_space_view() {return array_view1d<std::vector<SASC>>{(std::vector<SASC>*)anatomy.producing_space.data(), std::min<uint32_t>(c[BT::ProducerBlock], anatomy.producing_space.size())};}

    inline auto get_eating_space_view   () {return array_view1d<SASC>{(SASC*)anatomy.eating_space.data(),  std::min<size_t>(anatomy.eating_mask[c[BT::MouthBlock]-1], anatomy.eating_space.size())};}
    inline auto get_killing_space_view  () {return array_view1d<SASC>{(SASC*)anatomy.killing_space.data(), std::min<size_t>(anatomy.killer_mask[c[BT::KillerBlock]-1], anatomy.killing_space.size())};}

    void init_brain_type();

    void mutate_legacy(Anatomy &new_anatomy, lehmer64 &gen);
    void mutate_occ(Anatomy &new_anatomy, lehmer64 &gen, OrganismConstructionCode &new_occ);
    void mutate_mutation_rate(float &_anatomy_mutation_rate, lehmer64 &gen) const;
};


#endif //THELIFEENGINECPP_ORGANISM_H
