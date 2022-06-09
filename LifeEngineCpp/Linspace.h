//
// Created by spaceeye on 26.04.2022.
//

#ifndef THELIFEENGINECPP_LINSPACE_H
#define THELIFEENGINECPP_LINSPACE_H

#include <vector>
#include <cmath>

template<typename T>
std::vector<T> linspace(double start, double end, int num) {
    std::vector<T> linspaced;
    linspaced.reserve(num);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.template emplace_back(start);
        return linspaced;
    }

    //If range contains 0, then strange things happens, so I displace range by start to calculate delta and then subtract it from result.
    double displacement = 0;

    if (start <= 0) { displacement = std::abs(start) + 1;}

    start += displacement;
    end += displacement;

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num-1; ++i)
    {
        linspaced.template emplace_back(T(start + delta * i) - displacement);
    }
    linspaced.template emplace_back(end);

    return linspaced;
}

#endif //THELIFEENGINECPP_LINSPACE_H
