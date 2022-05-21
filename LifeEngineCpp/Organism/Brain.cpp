//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"

Brain::Brain(std::shared_ptr<Brain> & brain): mt(brain->mt), brain_type(brain->brain_type){
    copy_parents_table(brain->simple_action_table);
}

Brain::Brain(std::mt19937 *mt, BrainTypes brain_type): mt(mt), brain_type(brain_type) {}

BrainDecision Brain::get_random_action() {
    return static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 6)(*mt));
}


SimpleActionTable Brain::copy_parents_table(SimpleActionTable &parents_simple_action_table) {
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

SimpleActionTable Brain::mutate_action_table(SimpleActionTable &parents_simple_action_table, std::mt19937 &mt) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, 8)(mt));
    auto new_simple_action_table = copy_parents_table(parents_simple_action_table);

    auto new_decision = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));

    switch (mutate_type){
        case MouthBlock:    new_simple_action_table.MouthBlock    = new_decision;break;
        case ProducerBlock: new_simple_action_table.ProducerBlock = new_decision;break;
        case MoverBlock:    new_simple_action_table.MoverBlock    = new_decision;break;
        case KillerBlock:   new_simple_action_table.KillerBlock   = new_decision;break;
        case ArmorBlock:    new_simple_action_table.ArmorBlock    = new_decision;break;
        case EyeBlock:      new_simple_action_table.EyeBlock      = new_decision;break;
        case FoodBlock:     new_simple_action_table.FoodBlock     = new_decision;break;
        case WallBlock:     new_simple_action_table.WallBlock     = new_decision;break;
        default: break;
    }
    return new_simple_action_table;
}

SimpleActionTable Brain::get_random_action_table(std::mt19937 &mt) {
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

BrainDecision Brain::get_simple_action(std::vector<Observation> &observations_vector) {
    Observation * closest_observation;
    auto min_distance = INT32_MAX;
    auto observation_i = -1;

    for (int i = 0; i < observations_vector.size(); i++) {
        //if observation is blocked by something, then pass
        if (observations_vector[i].distance == 0) {
            continue;
        }
        if (min_distance < observations_vector[i].distance) {
            min_distance = observations_vector[i].distance;
            observation_i = i;
        }
    }
    //if there is no meaningful observations, then return random action;
    if (observation_i < 0) {return get_random_action();}

    return calculate_simple_action(observations_vector[observation_i]);
}

BrainDecision Brain::calculate_simple_action(Observation &observation) {
    auto action = SimpleDecision{};
    switch (observation.type) {
        case MouthBlock:    action = simple_action_table.MouthBlock;    break;
        case ProducerBlock: action = simple_action_table.ProducerBlock; break;
        case MoverBlock:    action = simple_action_table.MoverBlock;    break;
        case KillerBlock:   action = simple_action_table.KillerBlock;   break;
        case ArmorBlock:    action = simple_action_table.ArmorBlock;    break;
        case EyeBlock:      action = simple_action_table.EyeBlock;      break;
        case FoodBlock:     action = simple_action_table.FoodBlock;     break;
        case WallBlock:     action = simple_action_table.WallBlock;     break;
    }

    switch (action) {
        case SimpleDecision::DoNothing:
            return get_random_action();
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
    return get_random_action();
}

//TODO
BrainDecision Brain::get_decision(std::vector<Observation> &observation_vector) {
    switch (brain_type) {
        case BrainTypes::RandomActions:
            return get_random_action();
        case BrainTypes::SimpleBrain:
            return get_simple_action(observation_vector);
        case BrainTypes::BehaviourTreeBrain:
            break;
        case BrainTypes::NeuralNetworkBrain:
            break;
    }
    return get_random_action();
}

Brain * Brain::mutate() {
    auto new_brain = new Brain(mt, brain_type);
    new_brain->simple_action_table = mutate_action_table(simple_action_table, *mt);
    return new_brain;
}
