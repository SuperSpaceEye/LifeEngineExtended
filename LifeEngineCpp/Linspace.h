//
// Created by spaceeye on 26.04.2022.
//

#ifndef THELIFEENGINECPP_LINSPACE_H
#define THELIFEENGINECPP_LINSPACE_H

#include <vector>

template<typename T>
class Linspace {
public:
    std::vector<T> operator()(double start, double end, int num);
};

template<typename T>
std::vector<T> Linspace<T>::operator()(double start, double end, int num) {
    std::vector<T> linspaced;
    linspaced.reserve(num);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.template emplace_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.template emplace_back(start + delta * i);
    }
    linspaced.template emplace_back(end);

    return linspaced;
}


#endif //THELIFEENGINECPP_LINSPACE_H
