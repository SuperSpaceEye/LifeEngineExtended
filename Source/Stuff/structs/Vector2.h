//
// Created by spaceeye on 19.07.22.
//

#ifndef LIFEENGINEEXTENDED_VECTOR2_H
#define LIFEENGINEEXTENDED_VECTOR2_H

//operator no reference
#define OP_NR(op) Vector2 operator op(const Vector2 & o) {return {x op o.x, y op o.y};}
//simple operator no reference
#define SOP_NR(op) template<typename TT> Vector2 operator op(const TT & o) {return {x op o, y op o};}

#define OP_R(op) Vector2 & operator op(const Vector2 & o) {x op o.x; y op o.y; return *this;}
#define SOP_R(op) template<typename TT> Vector2 &operator op(const TT & o) {x op o; y op o; return *this;}

template <typename T>
struct Vector2 {
    T x;
    T y;
    constexpr Vector2(T x, T y): x(x), y(y) {}
    Vector2()=default;

    OP_NR(+)
    OP_NR(-)

    SOP_NR(+)
    SOP_NR(-)
    SOP_NR(*)
    SOP_NR(/)

    OP_R(+=)
    OP_R(-=)

    SOP_R(+=)
    SOP_R(-=)
    SOP_R(*=)
    SOP_R(/=)
};

#endif //LIFEENGINEEXTENDED_VECTOR2_H
