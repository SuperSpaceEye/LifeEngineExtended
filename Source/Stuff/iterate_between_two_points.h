//
// Created by spaceeye on 29.03.23.
//

#ifndef LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H
#define LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H

#include <vector>
#include <cmath>

#include "Stuff/structs/Vector2.h"

//https://gist.github.com/DavidMcLaughlin208/60e69e698e3858617c322d80a8f174e2
inline std::vector<Vector2<int>> iterate_between_two_points(Vector2<int> pos1, Vector2<int> pos2) {
    if (pos1.x == pos2.x && pos1.y == pos2.y) {return {pos1};}

    std::vector<Vector2<int>> points;
    int matrixX1 = pos1.x;
    int matrixY1 = pos1.y;
    int matrixX2 = pos2.x;
    int matrixY2 = pos2.y;

    int x_diff = matrixX1 - matrixX2;
    int y_diff = matrixY1 - matrixY2;
    bool x_diff_is_larger = std::abs(x_diff) > std::abs(y_diff);

    int x_modifier = x_diff < 0 ? 1 : -1;
    int y_modifier = y_diff < 0 ? 1 : -1;

    int longer_side_length  = std::max(std::abs(x_diff), std::abs(y_diff));
    int shorter_side_length = std::min(std::abs(x_diff), std::abs(y_diff));

    float slope = (shorter_side_length == 0 || longer_side_length == 0) ? 0 : ((float) (shorter_side_length) / (longer_side_length));

    int shorter_side_increase;
    for (int i = 1; i <= longer_side_length; i++) {
        shorter_side_increase = std::round(i * slope);
        int yIncrease, xIncrease;
        if (x_diff_is_larger) {
            xIncrease = i;
            yIncrease = shorter_side_increase;
        } else {
            yIncrease = i;
            xIncrease = shorter_side_increase;
        }
        int currentY = matrixY1 + (yIncrease * y_modifier);
        int currentX = matrixX1 + (xIncrease * x_modifier);
        points.emplace_back(currentX, currentY);
    }

    return points;
}

#endif //LIFEENGINEEXTENDED_ITERATE_BETWEEN_TWO_POINTS_H
