//
// Created by spaceeye on 19.07.22.
//

#ifndef LIFEENGINEEXTENDED_VECTOR2_H
#define LIFEENGINEEXTENDED_VECTOR2_H

template <typename T>
struct Vector2 {
    T x;
    T y;
    Vector2(T x, T y): x(x), y(y) {}
    Vector2()=default;
};

#endif //LIFEENGINEEXTENDED_VECTOR2_H
