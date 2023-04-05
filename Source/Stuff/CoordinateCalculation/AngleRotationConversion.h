//
// Created by spaceeye on 03.04.23.
//

#ifndef LIFEENGINEEXTENDED_ANGLEROTATIONCONVERSION_H
#define LIFEENGINEEXTENDED_ANGLEROTATIONCONVERSION_H

#include <cmath>

#include "Stuff/enums/Rotation.h"

template<typename T>
inline Rotation snap_to_closest_rotation(T angle) {
    // 0 deg - right, 90 deg - up, 180 deg - left, 270 deg - down
    return Rotation(u_int8_t(std::round(angle/(M_PI/2))-1)%4);
}

//percentage shows how much it is aligned with direction.
template<typename T>
inline std::pair<Rotation, T> snap_to_closest_rotation_with_percentage(T angle) {
    // 0 deg - right, 90 deg - up, 180 deg - left, 270 deg - down
    auto temp = angle/(M_PI/2);
    auto rounded = std::round(temp);
    T percentage = std::abs(std::abs(rounded - temp)-0.5)*2;
    return {Rotation(u_int8_t(rounded-1)%4), percentage};
}

template<typename T>
inline Rotation snap_to_second_closest_rotation(T angle) {
    auto closest_angle = std::round(angle/(M_PI/2))*(M_PI/2);
    auto pos = closest_angle - angle;
    return Rotation(uint32_t((int)snap_to_closest_rotation(closest_angle) + (pos < 0 ? 1 : -1))%4);
}

template<typename T>
inline std::pair<Rotation, T> snap_to_second_closest_rotation_with_percentage(T angle) {
    auto closest_angle = std::round(angle/(M_PI/2))*(M_PI/2);
    auto pos = closest_angle - angle;
    auto [closest, percentage] = snap_to_closest_rotation(closest_angle);
    return {Rotation(uint32_t((int)closest + (pos < 0 ? 1 : -1))%4), 1-percentage};
}


#endif //LIFEENGINEEXTENDED_ANGLEROTATIONCONVERSION_H
