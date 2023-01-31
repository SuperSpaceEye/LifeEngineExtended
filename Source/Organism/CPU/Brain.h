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

struct SimpleActionTable {
    //decision array
    //contains empty observation type which cannot be mutated
    std::array<SimpleDecision, NUM_WORLD_BLOCKS> da{
        // is computed at compile time
            []() constexpr {
                std::array<SimpleDecision, NUM_WORLD_BLOCKS> data{};
                for (int i = 0; i < NUM_WORLD_BLOCKS; i++) {
                    if (i+1 == int(BlockTypes::KillerBlock)) {
                        data[i] = SimpleDecision::GoAway;
                    } else if (i+1 == int(BlockTypes::FoodBlock)) {
                        data[i] = SimpleDecision::GoTowards;
                    } else {
                        data[i] = SimpleDecision::DoNothing;
                    }
                }
                return data;
            }()
    };
};

// "-1" - go away, "1" - go towards, "0" - neutral
struct WeightedActionTable {
    std::array<float, NUM_WORLD_BLOCKS> da{
            // is computed at compile time
            []() constexpr {
                std::array<float, NUM_WORLD_BLOCKS> data{};
                for (int i = 0; i < NUM_WORLD_BLOCKS; i++) {
                    if (i+1 == int(BlockTypes::KillerBlock)) {
                        data[i] = -1;
                    } else if (i+1 == int(BlockTypes::FoodBlock)) {
                        data[i] = 1;
                    } else {
                        data[i] = 0;
                    }
                }
                return data;
            }()
    };
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
    DecisionObservation get_weighted_action_discrete(std::vector<Observation> &observations_vector, int look_range,
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

    void set_brain(Brain brain);

    std::array<float, 4> get_weighted_direction(std::vector<Observation> &observations_vector, int look_range) const;

    std::pair<std::array<float, 4>, bool>
    get_global_weighted_direction(std::vector<Observation> &observations_vector, int look_range,
                                  Rotation organism_rotation) const;
};


#endif //THELIFEENGINECPP_BRAIN_H
