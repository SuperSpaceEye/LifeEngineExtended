//
// Created by spaceeye on 29.03.23.
//

#ifndef LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H
#define LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H

#include <vector>
#include <cmath>

#include "Stuff/structs/Vector2.h"

template<typename T>
Vector2<T> calculate_pos(const Vector2<T> &pos1, bool x_diff_is_larger, T x_modifier, T y_modifier, float slope,
                         T shorter_side_increase, int i);

//https://gist.github.com/DavidMcLaughlin208/60e69e698e3858617c322d80a8f174e2
template<typename T>
inline std::vector<Vector2<T>> iterate_between_two_points(Vector2<T> pos1, Vector2<T> pos2) {
    if (pos1.x == pos2.x && pos1.y == pos2.y) {return {pos1};}

    std::vector<Vector2<T>> points;

    T x_diff = pos1.x - pos2.x;
    T y_diff = pos1.x - pos2.y;
    bool x_diff_is_larger = std::abs(x_diff) > std::abs(y_diff);

    T x_modifier = x_diff < 0 ? 1 : -1;
    T y_modifier = y_diff < 0 ? 1 : -1;

    T longer_side_length  = std::max(std::abs(x_diff), std::abs(y_diff));
    T shorter_side_length = std::min(std::abs(x_diff), std::abs(y_diff));

    float slope = (shorter_side_length == 0 || longer_side_length == 0) ? 0 : ((float) (shorter_side_length) / (longer_side_length));

    T shorter_side_increase;
    for (int i = 1; i <= longer_side_length; i++) {
        Vector2<T> curr = calculate_pos(pos1, x_diff_is_larger, x_modifier, y_modifier, slope, shorter_side_increase,
                                        i);
        points.emplace_back(curr.x, curr.y);
    }

    return points;
}

template<typename T>
Vector2<T> calculate_pos(const Vector2<T> &pos1, bool x_diff_is_larger, T x_modifier, T y_modifier, float slope,
                         T shorter_side_increase, int i) {
    shorter_side_increase = std::round(i * slope);
    T yIncrease, xIncrease;
    if (x_diff_is_larger) {
        xIncrease = i;
        yIncrease = shorter_side_increase;
    } else {
        yIncrease = i;
        xIncrease = shorter_side_increase;
    }
    Vector2<T> curr = {
            pos1.y + (yIncrease * y_modifier),
            pos1.x + (xIncrease * x_modifier)
    };
    return curr;
}

#endif //LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H
