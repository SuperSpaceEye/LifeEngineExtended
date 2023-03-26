//
// Created by spaceeye on 12.03.23.
//

#ifndef LIFEENGINEEXTENDED_ANATOMYCOUNTERSMAP_H
#define LIFEENGINEEXTENDED_ANATOMYCOUNTERSMAP_H

#include "Stuff/structs/ConstMap.h"
#include "Stuff/enums/BlockTypes.hpp"

template <typename T, int len, std::string_view allowed[len]>
struct AnatomyCounters: public ConstMap<T, len, allowed> {
    template<std::size_t N>
    __attribute__((optimize("-O3"))) inline constexpr T & operator[](const char (&name)[N]) {
        return this->get(name);
    }

    __attribute__((optimize("-O3"))) inline constexpr T & operator[](const std::string_view name) {
        return this->get(name);
    }

    template<std::size_t N>
    __attribute__((optimize("-O3"))) inline constexpr const T & operator[](const char (&name)[N]) const {
        return this->cget(name);
    }

    __attribute__((optimize("-O3"))) inline constexpr const T & operator[](const std::string_view name) const {
        return this->cget(name);
    }

    __attribute__((optimize("-O3"))) inline constexpr T & operator[](const BlockTypes & type) {
        return this->data[(int)type-1];
    }

    __attribute__((optimize("-O3"))) inline constexpr const T & operator[](const BlockTypes & type) const {
        return this->data[(int)type-1];
    }
};

constexpr auto make_anatomy_counters(){
    return AnatomyCounters<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES>{};
};

#endif //LIFEENGINEEXTENDED_ANATOMYCOUNTERSMAP_H
