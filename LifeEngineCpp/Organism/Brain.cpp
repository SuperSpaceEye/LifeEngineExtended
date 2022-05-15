//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"

Brain::Brain(Brain &brain): mt(brain.mt), brain_type(brain.brain_type){}

Brain::Brain(std::mt19937 *mt, BrainTypes brain_type): mt(mt), brain_type(brain_type) {}

BrainDecision Brain::get_random_action() {
    return static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 6)(*mt));
}

//TODO
BrainDecision Brain::get_decision(std::vector<Observation> &observation_vector) {
    return get_random_action();
}
