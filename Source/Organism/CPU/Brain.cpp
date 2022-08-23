// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"
#include "ObservationStuff.h"

Brain::Brain(Brain & brain): brain_type(brain.brain_type), simple_action_table(copy_parents_table(brain.simple_action_table)) {}

Brain::Brain(const Brain &brain): brain_type(brain.brain_type), simple_action_table(copy_parents_table(brain.simple_action_table)) {}

Brain::Brain(BrainTypes brain_type): brain_type(brain_type) {}

void Brain::set_simple_action_table(Brain brain) {
    brain_type = brain.brain_type;
    simple_action_table = brain.simple_action_table;
}

DecisionObservation Brain::get_random_action(lehmer64 &mt) {
//    return static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 6)(gen));
    return DecisionObservation{static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 3)(mt)), Observation(), 0};
}


SimpleActionTable Brain::copy_parents_table(const SimpleActionTable &parents_simple_action_table) {
    auto simple_action_table = SimpleActionTable{};
    simple_action_table.MouthBlock    = parents_simple_action_table.MouthBlock;
    simple_action_table.ProducerBlock = parents_simple_action_table.ProducerBlock;
    simple_action_table.MoverBlock    = parents_simple_action_table.MoverBlock;
    simple_action_table.KillerBlock   = parents_simple_action_table.KillerBlock;
    simple_action_table.ArmorBlock    = parents_simple_action_table.ArmorBlock;
    simple_action_table.EyeBlock      = parents_simple_action_table.EyeBlock;
    simple_action_table.FoodBlock     = parents_simple_action_table.FoodBlock;
    simple_action_table.WallBlock     = parents_simple_action_table.WallBlock;
    return simple_action_table;
}

SimpleActionTable Brain::mutate_action_table(SimpleActionTable &parents_simple_action_table, lehmer64 &mt) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, 8)(mt));
    auto new_simple_action_table = copy_parents_table(parents_simple_action_table);

    auto new_decision = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));

    switch (mutate_type){
        case BlockTypes::MouthBlock:    new_simple_action_table.MouthBlock    = new_decision;break;
        case BlockTypes::ProducerBlock: new_simple_action_table.ProducerBlock = new_decision;break;
        case BlockTypes::MoverBlock:    new_simple_action_table.MoverBlock    = new_decision;break;
        case BlockTypes::KillerBlock:   new_simple_action_table.KillerBlock   = new_decision;break;
        case BlockTypes::ArmorBlock:    new_simple_action_table.ArmorBlock    = new_decision;break;
        case BlockTypes::EyeBlock:      new_simple_action_table.EyeBlock      = new_decision;break;
        case BlockTypes::FoodBlock:     new_simple_action_table.FoodBlock     = new_decision;break;
        case BlockTypes::WallBlock:     new_simple_action_table.WallBlock     = new_decision;break;
        default: break;
    }
    return new_simple_action_table;
}

SimpleActionTable Brain::get_random_action_table(lehmer64 &mt) {
    auto new_simple_action_table = SimpleActionTable{};
    new_simple_action_table.MouthBlock    = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.ProducerBlock = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.MoverBlock    = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.KillerBlock   = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.ArmorBlock    = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.EyeBlock      = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.FoodBlock     = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    new_simple_action_table.WallBlock     = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));
    return new_simple_action_table;
}

DecisionObservation Brain::get_simple_action(std::vector<Observation> &observations_vector, lehmer64 &mt) {
    auto min_distance = INT32_MAX;
    auto observation_i = -1;

    for (int i = 0; i < observations_vector.size(); i++) {
        //if observation is blocked by something, then pass
        if (observations_vector[i].distance == 0 || calculate_simple_action(observations_vector[i]) == BrainDecision::DoNothing) {
            continue;
        }
        if (observations_vector[i].distance < min_distance) {
            min_distance = observations_vector[i].distance;
            observation_i = i;
        }
    }
    //if there is no meaningful observations, then do nothing;
    if (observation_i < 0) {return DecisionObservation{BrainDecision::DoNothing, Observation{}};}

    return DecisionObservation{calculate_simple_action(observations_vector[observation_i]), observations_vector[observation_i]};
}

BrainDecision Brain::calculate_simple_action(Observation &observation) const {
    SimpleDecision action;
    switch (observation.type) {
        case BlockTypes::MouthBlock:    action = simple_action_table.MouthBlock;    break;
        case BlockTypes::ProducerBlock: action = simple_action_table.ProducerBlock; break;
        case BlockTypes::MoverBlock:    action = simple_action_table.MoverBlock;    break;
        case BlockTypes::KillerBlock:   action = simple_action_table.KillerBlock;   break;
        case BlockTypes::ArmorBlock:    action = simple_action_table.ArmorBlock;    break;
        case BlockTypes::EyeBlock:      action = simple_action_table.EyeBlock;      break;
        case BlockTypes::FoodBlock:     action = simple_action_table.FoodBlock;     break;
        case BlockTypes::WallBlock:     action = simple_action_table.WallBlock;     break;
        case BlockTypes::EmptyBlock:    action = SimpleDecision::DoNothing;         break;
        default: throw "unknown block";
    }

    switch (action) {
        case SimpleDecision::DoNothing:
            return BrainDecision::DoNothing;
        case SimpleDecision::GoAway:
            switch (observation.eye_rotation)
            {   //local movement
                case Rotation::UP:    return BrainDecision::MoveDown;
                case Rotation::LEFT:  return BrainDecision::MoveRight;
                case Rotation::DOWN:  return BrainDecision::MoveUp;
                case Rotation::RIGHT: return BrainDecision::MoveLeft;
            }
        case SimpleDecision::GoTowards:
            switch (observation.eye_rotation) {
                case Rotation::UP:    return BrainDecision::MoveUp;
                case Rotation::LEFT:  return BrainDecision::MoveLeft;
                case Rotation::DOWN:  return BrainDecision::MoveDown;
                case Rotation::RIGHT: return BrainDecision::MoveRight;
            }
    }
    throw "bad";
}

DecisionObservation Brain::get_decision(std::vector<Observation> &observation_vector, Rotation organism_rotation, lehmer64 &mt) {
    DecisionObservation action;
    switch (brain_type) {
        case BrainTypes::RandomActions:
            action = get_random_action(mt);
            break;
        case BrainTypes::SimpleBrain:
            action = get_simple_action(observation_vector, mt);
            break;
        case BrainTypes::BehaviourTreeBrain:
            break;
        case BrainTypes::NeuralNetworkBrain:
            break;
    }
    if (action.decision == BrainDecision::DoNothing) {
        return action;
    }

    uint_fast8_t new_int_action = static_cast<uint_fast8_t>(action.decision) + static_cast<uint_fast8_t>(organism_rotation);
    action.decision = static_cast<BrainDecision>(new_int_action%4);

    return action;
}

Brain Brain::mutate(lehmer64 &mt) {
    auto new_brain = Brain(brain_type);
    new_brain.simple_action_table = mutate_action_table(simple_action_table, mt);
    return new_brain;
}