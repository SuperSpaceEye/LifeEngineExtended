//
// Created by spaceeye on 03.04.23.
//

#ifndef LIFEENGINEEXTENDED_ANGLECONVERSION_H
#define LIFEENGINEEXTENDED_ANGLECONVERSION_H

#include <cmath>
#ifdef __DEBUG
#include <stdexcept>
#endif

#include "Stuff/enums/Rotation.h"
#include "Stuff/structs/Vector2.h"

template<typename T>
inline T rad_to_deg(T rad) {
#ifdef __DEBUG__
    if (rad * (180/M_PI) > 361. || rad * (180/M_PI) < 0) {
        throw std::logic_error("Degree shouldn't be > 360");
    }
#endif
    return rad * (180/M_PI);
}

template<typename T>
inline T deg_to_rad(T deg) {
#ifdef __DEBUG__
    if (deg > 361. || deg < 0) {
        throw std::logic_error("Degree shouldn't be > 360");
    }
#endif
    return deg * (M_PI/180);
}

inline double rotation_to_radians(Rotation rt) {
    return (int(rt)+1)*90*(M_PI/180);
}

#endif //LIFEENGINEEXTENDED_ANGLECONVERSION_H
