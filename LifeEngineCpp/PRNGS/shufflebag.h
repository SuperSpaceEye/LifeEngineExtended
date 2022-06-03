//
// Created by spaceeye on 03.06.22.
//

#ifndef THELIFEENGINECPP_SHUFFLEBAG_H
#define THELIFEENGINECPP_SHUFFLEBAG_H

//https://stackoverflow.com/questions/1046714/what-is-a-good-random-number-generator-for-a-game

#include <cstdint>
#include <utility>
#include <vector>

class shufflebag {
    std::vector<uint64_t> states;
    int position = 1;
    int sequence_length = 0;
    uint64_t min_val = 0;
    uint64_t max_val = UINT64_MAX;
public:
    using result_type = uint64_t;

    template<class Generator>
    shufflebag(Generator &gen, int sequence_length) {
        initialize(gen, sequence_length);
        min_val = gen.min();
        max_val = gen.max();
    }

    explicit shufflebag(std::vector<uint64_t> &states) {
        initialize(states);
    }

    shufflebag()=default;

    constexpr result_type min() const { return min_val; }
    constexpr result_type max() const { return max_val; }
    result_type operator()() {
        position++;
        if (position > sequence_length) {
            position = 1;
        }

        return states[position-1];
    }

    template<class Generator>
    void initialize(Generator &gen, int sequence_length) {
        states.resize(sequence_length);
        for (int i = 0; i < sequence_length; i++) {
            states.template emplace_back(gen());
        }
        position = 1;
        this->sequence_length = sequence_length;
    }

    void initialize(std::vector<uint64_t> &states) {
        this->states = std::move(states);
        position = 1;
        sequence_length = this->states.size();
    }
};

#endif //THELIFEENGINECPP_SHUFFLEBAG_H
