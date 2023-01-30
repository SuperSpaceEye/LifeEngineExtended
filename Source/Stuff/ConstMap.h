//
// Created by spaceeye on 31.01.23.
//

#ifndef LIFEENGINEEXTENDED_CONSTMAP_H
#define LIFEENGINEEXTENDED_CONSTMAP_H

#include <array>
#include <string_view>

template <typename T, int len, std::string_view allowed[len]>
struct ConstMap {
    std::array<T, len> data;

    template<std::size_t N>
    constexpr T & get(const char (&name)[N]) {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    constexpr T & get(const std::string_view name) {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    template<std::size_t N>
    constexpr const T & cget(const char (&name)[N]) const {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    constexpr const T & cget(const std::string_view name) const {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    template<std::size_t N>
    constexpr T & operator[](const char (&name)[N]) {
        return get(name);
    }

    constexpr T & operator[](const std::string_view name) {
        return get(name);
    }

    template<std::size_t N>
    constexpr const T & operator[](const char (&name)[N]) const {
        return cget(name);
    }

    constexpr const T & operator[](const std::string_view name) const {
        return cget(name);
    }
};

#endif //LIFEENGINEEXTENDED_CONSTMAP_H
