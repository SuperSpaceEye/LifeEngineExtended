// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 12.05.2022.
//

#ifndef THELIFEENGINECPP_BRAIN_H
#define THELIFEENGINECPP_BRAIN_H

#include <cstdint>
#include <vector>
#include <random>
#include <memory>
#include "../../PRNGS/lehmer64.h"

#include "../../Stuff/BlockTypes.hpp"
#include "Rotation.h"
#include "ObservationStuff.h"
#include "../../Containers/CPU/SimulationParameters.h"

enum class BrainDecision {
    //Movement and rotation from organism viewpoint.
    MoveUp,
    MoveLeft,
    MoveDown,
    MoveRight,

    RotateLeft,
    RotateRight,
    Flip,

    DoNothing,

    TryProduceChild,
};

//Maybe for later
enum class BrainTypes {
    RandomActions,
    // chooses the closest observation to an editor_organism, and acts upon it. If do nothing, then returns random action.
    // If no meaningful action, then returns random action.
    SimpleBrain,
    WeightedBrain,
    //TODO will try to implement in the future
    //https://gamedev.stackexchange.com/questions/51693/difference-between-decision-trees-behavior-trees-for-game-ai
    //https://www.behaviortree.dev/
    //https://github.com/BehaviorTree/BehaviorTree.CPP
    BehaviourTreeBrain,
    //I will need a c++ torch or something.
    NeuralNetworkBrain,
};

enum class SimpleDecision {
    DoNothing,
    GoAway,
    GoTowards
};

struct DecisionObservation {
    BrainDecision decision = BrainDecision::MoveUp;
    Observation observation = Observation{};
    int time = 0;
};

//TODO remember convert
struct SimpleActionTable {
    SimpleDecision MouthBlock    = SimpleDecision::DoNothing;
    SimpleDecision ProducerBlock = SimpleDecision::DoNothing;
    SimpleDecision MoverBlock    = SimpleDecision::DoNothing;
    SimpleDecision KillerBlock   = SimpleDecision::GoAway;
    SimpleDecision ArmorBlock    = SimpleDecision::DoNothing;
    SimpleDecision EyeBlock      = SimpleDecision::DoNothing;
    SimpleDecision FoodBlock     = SimpleDecision::GoTowards;
    SimpleDecision WallBlock     = SimpleDecision::DoNothing;
};

// "-1" - go away, "1" - go towards, "0" - neutral
struct WeightedActionTable {
    float MouthBlock    = 0;
    float ProducerBlock = 0;
    float MoverBlock    = 0;
    float KillerBlock   = -1;
    float ArmorBlock    = 0;
    float EyeBlock      = 0;
    float FoodBlock     = 1;
    float WallBlock     = 0;
};

struct BrainWeightedDecision {
    BrainDecision decision = BrainDecision::MoveUp;
    float weight = 0;
};

class Brain {
private:
    static SimpleActionTable mutate_simple_action_table(SimpleActionTable &parents_simple_action_table, lehmer64 &mt);
    static WeightedActionTable
    mutate_weighted_action_table(WeightedActionTable &parent_action_table, lehmer64 &mt, SimulationParameters &sp);

    DecisionObservation get_simple_action(std::vector<Observation> &observations_vector);
    DecisionObservation get_weighted_action(std::vector<Observation> &observations_vector, int look_range,
                                      float threshold_move);

    BrainDecision calculate_simple_action(Observation &observation) const;
    BrainWeightedDecision calculate_weighted_action(Observation &observation, int look_range) const;
public:
    Brain()=default;
    Brain(Brain & brain);
    Brain(const Brain &brain);
    explicit Brain(BrainTypes brain_type);
    Brain(Brain&&)=default;
    Brain & operator=(const Brain & brain)=default;

    SimpleActionTable simple_action_table;
    WeightedActionTable weighted_action_table;

    BrainTypes brain_type;

    static DecisionObservation get_random_action(lehmer64 &mt);
    DecisionObservation get_decision(std::vector<Observation> &observation_vector, Rotation organism_rotation, lehmer64 &mt,
                                     int look_range, float threshold_move);

    void convert_simple_to_weighted();
    void convert_weighted_to_simple(float threshold_move);


    Brain mutate(lehmer64 &mt, SimulationParameters sp);

    void set_simple_action_table(Brain brain);
};


#endif //THELIFEENGINECPP_BRAIN_H
