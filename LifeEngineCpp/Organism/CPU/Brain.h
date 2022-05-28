//
// Created by spaceeye on 12.05.2022.
//

#ifndef THELIFEENGINECPP_BRAIN_H
#define THELIFEENGINECPP_BRAIN_H

#include <cstdint>
#include <vector>
#include <random>
#include <memory>
#include <boost/random.hpp>

#include "../../BlockTypes.hpp"
#include "Rotation.h"

enum class BrainDecision {
    //Movement and rotation from organism viewpoint.
    MoveUp,
    MoveLeft,
    MoveDown,
    MoveRight,

    RotateLeft,
    RotateRight,
    Flip,

    //TODO will not be used right now, but for more complex brains it will be.
    TryProduceChild,
};

//Maybe for later
enum class BrainTypes {
    RandomActions,
    // chooses the closest observation to an organism, and acts upon it. If do nothing, then returns random action.
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

struct Observation {
    BlockTypes type;
    int32_t distance;
    //local rotation
    Rotation eye_rotation;
};

enum class SimpleDecision {
    DoNothing,
    GoAway,
    GoTowards
};

struct SimpleActionTable {
    SimpleDecision MouthBlock;
    SimpleDecision ProducerBlock;
    SimpleDecision MoverBlock;
    SimpleDecision KillerBlock;
    SimpleDecision ArmorBlock;
    SimpleDecision EyeBlock;
    SimpleDecision FoodBlock;
    SimpleDecision WallBlock;
};

class Brain {
private:
    BrainDecision get_random_action(boost::mt19937 &mt);

    SimpleActionTable simple_action_table;
    static SimpleActionTable copy_parents_table(SimpleActionTable & parents_simple_action_table);
    static SimpleActionTable mutate_action_table(SimpleActionTable &parents_simple_action_table, boost::mt19937 &mt);
    static SimpleActionTable get_random_action_table(boost::mt19937 &mt);
    BrainDecision get_simple_action(std::vector<Observation> & observations_vector, boost::mt19937 &mt);
    BrainDecision calculate_simple_action(Observation & observation, boost::mt19937 &mt);
public:
    Brain()=default;
    explicit Brain(std::shared_ptr<Brain> & brain);
    Brain(BrainTypes brain_type);

    BrainTypes brain_type;


    BrainDecision get_decision(std::vector<Observation> &observation_vector, boost::mt19937 &mt);

    Brain * mutate(boost::mt19937 &mt);

};


#endif //THELIFEENGINECPP_BRAIN_H
