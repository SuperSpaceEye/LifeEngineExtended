//
// Created by spaceeye on 12.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTSENUMS_H
#define LIFEENGINEEXTENDED_WORLDEVENTSENUMS_H

enum class NodeType {
    ChangeValue,
    Conditional,
};
enum class ChangeValueMode {
    Step,
    Linear,
    IncreaseBy,
    DecreaseBy,
    MultiplyBy,
    DivideBy,
//    Exponential
//    Logarithmic
};
enum class ChangeTypes {
    NONE,
    INT32,
    FLOAT
};
enum class ClampModes {
    NoClamp,
    ClampMinValue,
    ClampMaxValue,
    ClampMinMaxValues
};
enum class ConditionalMode {
    MoreOrEqual,
    LessOrEqual,
};
enum class ConditionalTypes {
    NONE,
    DOUBLE,
    INT64,
};
#endif //LIFEENGINEEXTENDED_WORLDEVENTSENUMS_H
