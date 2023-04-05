//
// Created by spaceeye on 03.04.23.
//

#ifndef LIFEENGINEEXTENDED_GET_MAX_COORDINATE_H
#define LIFEENGINEEXTENDED_GET_MAX_COORDINATE_H

#include <cmath>

#include "Stuff/structs/Vector2.h"
#include "Stuff/enums/Rotation.h"

namespace {
    template<typename T>
    inline Vector2<T> unit_vector(double angle) {
        return Vector2<T>{(T) std::cos(angle), (T) std::sin(angle)};
    }

    template<typename T>
    inline Vector2<T> complex_mult(Vector2<T> a, Vector2<T> b) {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
}

// calculates coordinate with center_pos as axis center at a distance with an angle in radians.
// M_PI is added to angle because simulation grid has 0,0 in upper left corner.
template<typename T>
inline Vector2<T> get_max_coordinate(Vector2<T> center_pos, double angle, T distance) {
    return complex_mult({distance, 0}, unit_vector<T>(angle+M_PI)) + center_pos;
}

#endif //LIFEENGINEEXTENDED_GET_MAX_COORDINATE_H
