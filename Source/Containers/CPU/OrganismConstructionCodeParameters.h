//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEPARAMETERS_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEPARAMETERS_H

#include <random>
#include <vector>

//https://github.com/DavidPal/discrete-distribution

struct OCCParameters {
    bool uniform_mutation_distribution = true;
    std::array<int, 5> mutation_type_weights{1, 1, 1, 1, 1};
    std::discrete_distribution<int> mutation_discrete_distribution{mutation_type_weights.begin(), mutation_type_weights.end()};

    bool uniform_group_size_distribution = true;
    int max_group_size = 5;
    std::vector<int> group_size_weights{1, 1, 1, 1, 1};
    std::discrete_distribution<int> group_size_discrete_distribution{group_size_weights.begin(), group_size_weights.end()};

    bool uniform_occ_instructions_mutation = true;
    std::vector<int> occ_instructions_mutation_weights{1};
    std::discrete_distribution<int> occ_instructions_mutation_discrete_distribution{occ_instructions_mutation_weights.begin(), occ_instructions_mutation_weights.end()};

    bool uniform_move_distance = true;
    int max_distance = 5;
    std::vector<int> move_distance_mutation_weights{1, 1, 1, 1, 1};
    std::discrete_distribution<int> move_distance_mutation_discrete_distribution{move_distance_mutation_weights.begin(), move_distance_mutation_weights.end()};
};

#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODEPARAMETERS_H
