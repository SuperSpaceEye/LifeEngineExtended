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
    SimpleDecision MouthBlock    = SimpleDecision::DoNothing;
    SimpleDecision ProducerBlock = SimpleDecision::DoNothing;
    SimpleDecision MoverBlock    = SimpleDecision::DoNothing;
    SimpleDecision KillerBlock   = SimpleDecision::GoAway;
    SimpleDecision ArmorBlock    = SimpleDecision::DoNothing;
    SimpleDecision EyeBlock      = SimpleDecision::DoNothing;
    SimpleDecision FoodBlock     = SimpleDecision::GoTowards;
    SimpleDecision WallBlock     = SimpleDecision::DoNothing;
};

class Brain {
private:
    static SimpleActionTable copy_parents_table(SimpleActionTable & parents_simple_action_table);
    static SimpleActionTable mutate_action_table(SimpleActionTable &parents_simple_action_table, lehmer64 &mt);
    static SimpleActionTable get_random_action_table(lehmer64 &mt);
    DecisionObservation get_simple_action(std::vector<Observation> & observations_vector, lehmer64 &mt);
    BrainDecision calculate_simple_action(Observation &observation) const;
public:
    Brain()=default;
    explicit Brain(std::shared_ptr<Brain> & brain);
    Brain(BrainTypes brain_type);

    SimpleActionTable simple_action_table;

    BrainTypes brain_type;

    static DecisionObservation get_random_action(lehmer64 &mt);
    DecisionObservation get_decision(std::vector<Observation> &observation_vector, Rotation organism_rotation, lehmer64 &mt);

    Brain * mutate(lehmer64 &mt);

};


#endif //THELIFEENGINECPP_BRAIN_H
