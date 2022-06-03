//
// Created by spaceeye on 03.06.22.
//

#ifndef THELIFEENGINECPP_XORSHF96_H
#define THELIFEENGINECPP_XORSHF96_H

#include <cstdint>

//https://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c

class xorshf96 {
public:
    using result_type = uint64_t;
    xorshf96(result_type x, result_type y, result_type z):x(x), y(y), z(z) {}
    xorshf96()=default;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }
    result_type operator()() {//period 2^96-1
        uint64_t t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return z;
    }
private:
    result_type x=123456789, y=362436069, z=521288629;
};

#endif //THELIFEENGINECPP_XORSHF96_H
