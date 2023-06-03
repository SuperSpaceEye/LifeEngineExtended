//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_COMMON_CONCEPTS_H
#define LIFEENGINEEXTENDED_COMMON_CONCEPTS_H

#include "includes/std.h"
#include <concepts>

namespace Concepts {
    template<typename T>
    concept StringLike = std::is_convertible_v<T, std::string_view>;
}

#endif //LIFEENGINEEXTENDED_COMMON_CONCEPTS_H
