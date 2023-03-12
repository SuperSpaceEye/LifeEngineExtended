// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 12.05.2022.
//

#include "Brain.h"

Brain::Brain(Brain & brain): brain_type(brain.brain_type),
simple_action_table(SimpleActionTable{brain.simple_action_table}), weighted_action_table(brain.weighted_action_table) {}

Brain::Brain(const Brain &brain): brain_type(brain.brain_type),
simple_action_table(SimpleActionTable{brain.simple_action_table}), weighted_action_table(brain.weighted_action_table) {}

Brain::Brain(BrainTypes brain_type): brain_type(brain_type) {}

void Brain::set_brain(const Brain& brain) {
    brain_type = brain.brain_type;
    simple_action_table = brain.simple_action_table;
    weighted_action_table = brain.weighted_action_table;
}

DecisionObservation Brain::get_random_action(lehmer64 &mt) {
    return DecisionObservation{static_cast<BrainDecision>(std::uniform_int_distribution<int>(0, 3)(mt)), Observation(), 0};
}


SimpleActionTable Brain::mutate_simple_action_table(const SimpleActionTable &parents_simple_action_table, lehmer64 &mt) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, NUM_WORLD_BLOCKS-1)(mt));
    auto new_simple_action_table = SimpleActionTable{parents_simple_action_table};

    auto new_decision = static_cast<SimpleDecision>(std::uniform_int_distribution<int>(0, 2)(mt));

    new_simple_action_table.da[int(mutate_type)] = new_decision;

    return new_simple_action_table;
}

WeightedActionTable Brain::mutate_weighted_action_table(const WeightedActionTable &parent_action_table, lehmer64 &mt, SimulationParameters &sp) {
    auto mutate_type = static_cast<BlockTypes>(std::uniform_int_distribution<int>(1, NUM_WORLD_BLOCKS-1)(mt));
    auto new_weighted_action_table = WeightedActionTable{parent_action_table};

    float modif = sp.weighted_brain_mutation_step * (std::uniform_int_distribution<int>(0, 1)(mt) ? 1. : -1.);

    float * value = &new_weighted_action_table.da[int(mutate_type)];

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

BrainDecision Brain::calculate_simple_action(const Observation &observation) const {
    SimpleDecision action = simple_action_table.da[int(observation.type)];


    if (action == SimpleDecision::DoNothing) {return BrainDecision::DoNothing;}

#ifdef __DEBUG__
    if ((int)observation.eye_rotation < 0 || (int)observation.eye_rotation >= 4) {
        throw std::runtime_error("");
    }
#endif

    return std::array<std::array<BrainDecision, 4>, 2> {
            std::array<BrainDecision, 4>{
            BrainDecision::MoveDown,
            BrainDecision::MoveRight,
            BrainDecision::MoveUp,
            BrainDecision::MoveLeft
        }, std::array<BrainDecision, 4>{
            BrainDecision::MoveUp,
            BrainDecision::MoveLeft,
            BrainDecision::MoveDown,
            BrainDecision::MoveRight
        }
    }[int(action)-1][int(observation.eye_rotation)];
}

std::array<float, 4> Brain::get_weighted_direction(std::vector<Observation> &observations_vector,
                                                   int look_range) const {
    //up, left, down, right
    std::array<float, 4> weighted_directions{0, 0, 0, 0};

    for (auto & observation: observations_vector) {
        if (observation.distance == 0) {continue;}

        auto wd = calculate_weighted_action(observation, look_range);
        weighted_directions[static_cast<int>(wd.decision)] += wd.weight;
    }
    return weighted_directions;
}

inline Rotation get_global_rotation(Rotation rotation1, Rotation rotation2) {
    uint_fast8_t new_int_rotation = static_cast<uint_fast8_t>(rotation1) + static_cast<uint_fast8_t>(rotation2);
    return static_cast<Rotation>(new_int_rotation%4);
}

std::pair<std::array<float, 4>, bool>
Brain::get_global_weighted_direction(std::vector<Observation> &observations_vector, int look_range,
                                     Rotation organism_rotation) const {
    //up, left, down, right
    std::array<float, 4> weighted_directions{0, 0, 0, 0};
    bool had_observation = false;

    for (auto & observation: observations_vector) {
        if (observation.distance == 0) {continue;}

        auto wd = calculate_weighted_action(observation, look_range);

        weighted_directions[static_cast<int>(get_global_rotation((Rotation)wd.decision, organism_rotation))] += wd.weight;
        if (wd.weight > 0) had_observation = true;
    }
    return {weighted_directions, had_observation};
}

DecisionObservation Brain::get_weighted_action_discrete(std::vector<Observation> &observations_vector, int look_range, float threshold_move) {
    std::array<float, 4> weighted_directions = get_weighted_direction(observations_vector, look_range);

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

BrainWeightedDecision Brain::calculate_weighted_action(const Observation &observation, int look_range) const{
    float distance_modifier = (float)std::abs(observation.distance - look_range - 1) / look_range;
    float weight = weighted_action_table.da[int(observation.type)];

    weight *= distance_modifier;

    SimpleDecision action = weight > 0 ? SimpleDecision::GoTowards : SimpleDecision::GoAway;

#ifdef __DEBUG__
    if ((int)observation.eye_rotation < 0 || (int)observation.eye_rotation >= 4) {
        throw std::runtime_error("");
    }
#endif

    auto decision = std::array<std::array<BrainDecision, 4>, 2> {
            std::array<BrainDecision, 4>{
                    BrainDecision::MoveDown,
                    BrainDecision::MoveRight,
                    BrainDecision::MoveUp,
                    BrainDecision::MoveLeft
            }, std::array<BrainDecision, 4>{
                    BrainDecision::MoveUp,
                    BrainDecision::MoveLeft,
                    BrainDecision::MoveDown,
                    BrainDecision::MoveRight
            }}[(int)action-1][(int)observation.eye_rotation];

    return BrainWeightedDecision{decision, weight};
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
            action = get_weighted_action_discrete(observation_vector, look_range, threshold_move);
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
            new_brain.brain_type = BrainTypes::SimpleBrain;
            break;
        case BrainTypes::WeightedBrain:
            new_brain.weighted_action_table = mutate_weighted_action_table(weighted_action_table, mt, sp);
            new_brain.brain_type = BrainTypes::WeightedBrain;
            break;
    }
    return new_brain;
}

void Brain::convert_simple_to_weighted() {
    auto & sda = simple_action_table.da;
    auto & wda = weighted_action_table.da;

    for (int i = 0; i < NUM_WORLD_BLOCKS; i++) {
        switch (sda[i]) {
            case SimpleDecision::DoNothing: wda[i] = 0; break;
            case SimpleDecision::GoAway:    wda[i] = -1; break;
            case SimpleDecision::GoTowards: wda[i] = 1; break;
        }
    }

    brain_type = BrainTypes::WeightedBrain;
}

void Brain::convert_weighted_to_simple(float threshold_move) {
    auto & sda = simple_action_table.da;
    auto & wda = weighted_action_table.da;

    for (int i = 0; i < NUM_WORLD_BLOCKS; i++) {
        auto tw = wda[i];

        if (std::abs(tw) < threshold_move) {sda[i] = SimpleDecision::DoNothing;}
        else if (tw > 0) {sda[i] = SimpleDecision::GoTowards;}
        else {sda[i] = SimpleDecision::GoAway;}
    }
    brain_type = BrainTypes::SimpleBrain;
}