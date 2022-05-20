//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"

Brain::Brain(std::shared_ptr<Brain> & brain): mt(brain->mt), brain_type(brain->brain_type){}

Brain::Brain(std::mt19937 *mt, BrainTypes brain_type): mt(mt), brain_type(brain_type) {}

BrainDecision Brain::get_random_action() {

    return static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 6)(*mt));
}

//TODO
BrainDecision Brain::get_decision(std::vector<Observation> &observation_vector) {
    switch (brain_type) {
        case BrainTypes::RandomActions:
            return get_random_action();
        case BrainTypes::SimpleBrain:
            break;
        case BrainTypes::BehaviourTreeBrain:
            break;
        case BrainTypes::NeuralNetworkBrain:
            break;
    }
    return get_random_action();
}

Brain * Brain::mutate() {
    auto new_brain = new Brain(mt, BrainTypes::RandomActions);
    return new_brain;
}
