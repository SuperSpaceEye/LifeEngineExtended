//
// Created by spaceeye on 12.05.2022.
//

#ifndef THELIFEENGINECPP_BRAIN_H
#define THELIFEENGINECPP_BRAIN_H

#include <cstdint>
#include <vector>
#include <random>
#include <memory>

#include "../BlockTypes.h"

enum class BrainDecision {
    //Movement
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
};

class Brain {
private:
    BrainDecision get_random_action();
    BrainDecision get_simple_action(std::vector<Observation> & observations_vector);
public:
    Brain()=default;
    explicit Brain(std::shared_ptr<Brain> & brain);
    Brain(std::mt19937 * mt, BrainTypes brain_type);

    std::mt19937 * mt;
    BrainTypes brain_type;

    BrainDecision get_decision(std::vector<Observation> & observation_vector);

    Brain * mutate();

};


#endif //THELIFEENGINECPP_BRAIN_H
