// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"
#include "ObservationStuff.h"

Brain::Brain(Brain & brain): brain_type(brain.brain_type),
simple_action_table(SimpleActionTable{brain.simple_action_table}), weighted_action_table(brain.weighted_action_table) {}

Brain::Brain(const Brain &brain): brain_type(brain.brain_type),
simple_action_table(SimpleActionTable{brain.simple_action_table}), weighted_action_table(brain.weighted_action_table) {}

Brain::Brain(BrainTypes brain_type): brain_type(brain_type) {}

void Brain::set_simple_action_table(Brain brain) {
    brain_type = brain.brain_type;
    simple_action_table = brain.simple_action_table;
}

DecisionObservation Brain::get_random_action(lehmer64 &mt) {
    return DecisionObservation{static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 3)(mt)), Observation(), 0};
}


SimpleActionTable Brain::mutate_simple_action_table(SimpleActionTable &parents_simple_action_table, lehmer64 &mt) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, 8)(mt));
    auto new_simple_action_table = SimpleActionTable{parents_simple_action_table};

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

WeightedActionTable Brain::mutate_weighted_action_table(WeightedActionTable &parent_action_table, lehmer64 &mt, SimulationParameters &sp) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, 8)(mt));
    auto new_weighted_action_table = WeightedActionTable{parent_action_table};

    float modif = sp.weighted_brain_mutation_step * (std::uniform_int_distribution<int>(0, 1)(mt) ? 1. : -1.);

    float * value;

    switch (mutate_type){
        case BlockTypes::MouthBlock:    value = &new_weighted_action_table.MouthBlock    ;break;
        case BlockTypes::ProducerBlock: value = &new_weighted_action_table.ProducerBlock ;break;
        case BlockTypes::MoverBlock:    value = &new_weighted_action_table.MoverBlock    ;break;
        case BlockTypes::KillerBlock:   value = &new_weighted_action_table.KillerBlock   ;break;
        case BlockTypes::ArmorBlock:    value = &new_weighted_action_table.ArmorBlock    ;break;
        case BlockTypes::EyeBlock:      value = &new_weighted_action_table.EyeBlock      ;break;
        case BlockTypes::FoodBlock:     value = &new_weighted_action_table.FoodBlock     ;break;
        case BlockTypes::WallBlock:     value = &new_weighted_action_table.WallBlock     ;break;
        default: break;
    }

    if (modif > 0) {
        *value = std::min<float>(1.,  *value+modif);
    } else {
        *value = std::max<float>(-1., *value+modif);
    }

    return new_weighted_action_table;
}


DecisionObservation Brain::get_simple_action(std::vector<Observation> &observations_vector) {
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

DecisionObservation Brain::get_weighted_action(std::vector<Observation> &observations_vector, int look_range, float threshold_move) {
    //up, left, down, right
    std::array<float, 4> weighted_directions{0, 0, 0, 0};

    for (auto & observation: observations_vector) {
        if (observation.distance == 0) {continue;}

        auto wd = calculate_weighted_action(observation, look_range);
        weighted_directions[static_cast<int>(wd.decision)] += wd.weight;
    }

    float max_weight = 0;
    int direction = 0;

    for (int i = 0; i < weighted_directions.size(); i++) {
        if (std::abs(weighted_directions[i]) > std::abs(max_weight)) {
            max_weight = weighted_directions[i];
            direction = i;
        }
    }

    if (std::abs(max_weight) < threshold_move) {return DecisionObservation{BrainDecision::DoNothing, observations_vector[0], 0};}

    return DecisionObservation{static_cast<BrainDecision>(direction), observations_vector[0], 0};
}

BrainWeightedDecision Brain::calculate_weighted_action(Observation &observation, int look_range) const{
    float distance_modifier = (float)std::abs(observation.distance - look_range - 1) / look_range;
    float weight;

    switch (observation.type) {
        case BlockTypes::EmptyBlock:    weight = 0; break;
        case BlockTypes::MouthBlock:    weight = weighted_action_table.MouthBlock; break;
        case BlockTypes::ProducerBlock: weight = weighted_action_table.ProducerBlock; break;
        case BlockTypes::MoverBlock:    weight = weighted_action_table.MoverBlock; break;
        case BlockTypes::KillerBlock:   weight = weighted_action_table.KillerBlock; break;
        case BlockTypes::ArmorBlock:    weight = weighted_action_table.ArmorBlock; break;
        case BlockTypes::EyeBlock:      weight = weighted_action_table.EyeBlock; break;
        case BlockTypes::FoodBlock:     weight = weighted_action_table.FoodBlock; break;
        case BlockTypes::WallBlock:     weight = weighted_action_table.WallBlock; break;
    }

    weight *= distance_modifier;

    SimpleDecision action = weight > 0 ? SimpleDecision::GoTowards : SimpleDecision::GoAway;
    switch (action) {
        case SimpleDecision::GoAway:
            switch (observation.eye_rotation)
            {   //local movement
                case Rotation::UP:    return BrainWeightedDecision{BrainDecision::MoveDown, weight};
                case Rotation::LEFT:  return BrainWeightedDecision{BrainDecision::MoveRight, weight};
                case Rotation::DOWN:  return BrainWeightedDecision{BrainDecision::MoveUp, weight};
                case Rotation::RIGHT: return BrainWeightedDecision{BrainDecision::MoveLeft, weight};
            }
        case SimpleDecision::GoTowards:
            switch (observation.eye_rotation) {
                case Rotation::UP:    return BrainWeightedDecision{BrainDecision::MoveUp, weight};
                case Rotation::LEFT:  return BrainWeightedDecision{BrainDecision::MoveLeft, weight};
                case Rotation::DOWN:  return BrainWeightedDecision{BrainDecision::MoveDown, weight};
                case Rotation::RIGHT: return BrainWeightedDecision{BrainDecision::MoveRight, weight};
            }
    }
}


DecisionObservation
Brain::get_decision(std::vector<Observation> &observation_vector, Rotation organism_rotation, lehmer64 &mt,
                    int look_range, float threshold_move) {
    DecisionObservation action;
    switch (brain_type) {
        case BrainTypes::RandomActions:
            action = get_random_action(mt);
            break;
        case BrainTypes::SimpleBrain:
            action = get_simple_action(observation_vector);
            break;
        case BrainTypes::WeightedBrain:
            action = get_weighted_action(observation_vector, look_range, threshold_move);
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

Brain Brain::mutate(lehmer64 &mt, SimulationParameters sp) {
    auto new_brain = Brain(brain_type);
    switch (brain_type) {
        case BrainTypes::SimpleBrain:
            new_brain.simple_action_table = mutate_simple_action_table(simple_action_table, mt);
            break;
        case BrainTypes::WeightedBrain:
            new_brain.weighted_action_table = mutate_weighted_action_table(weighted_action_table, mt, sp);
            break;
    }
    return new_brain;
}