//
// Created by spaceeye on 03.06.22.
//

/**
    * D. H. Lehmer, Mathematical methods in large-scale computing units.
    * Proceedings of a Second Symposium on Large Scale Digital Calculating
    * Machinery;
    * Annals of the Computation Laboratory, Harvard Univ. 26 (1951), pp. 141-146.
    *
    * P L'Ecuyer,  Tables of linear congruential generators of different sizes and
    * good lattice structure. Mathematics of Computation of the American
    * Mathematical
    * Society 68.225 (1999): 249-260.
*/

//https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/

#ifndef THELIFEENGINECPP_LEHMER64_H
#define THELIFEENGINECPP_LEHMER64_H

#include <cstdint>
#include "splitmix64.h"

class lehmer64 {
    __uint128_t g_lehmer64_state = 0;
public:
    using result_type = uint64_t;
    lehmer64()=default;
    lehmer64(uint64_t seed) { set_seed(seed); }
    void set_seed(uint64_t seed) {
        g_lehmer64_state = (((__uint128_t)splitmix64_stateless(seed)) << 64) +
                           splitmix64_stateless(seed + 1);
    }
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }
    result_type operator()() {
        g_lehmer64_state *= UINT64_C(0xda942042e4dd58b5);
        return g_lehmer64_state >> 64;
    }
};

#endif //THELIFEENGINECPP_LEHMER64_H
