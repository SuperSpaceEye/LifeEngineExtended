// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

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
    __attribute__((optimize("-O3"))) inline constexpr T & get(const char (&name)[N]) {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    __attribute__((optimize("-O3"))) inline constexpr T & get(const std::string_view name) {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    template<std::size_t N>
    __attribute__((optimize("-O3"))) inline constexpr const T & cget(const char (&name)[N]) const {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    __attribute__((optimize("-O3"))) inline constexpr const T & cget(const std::string_view name) const {
        for (int i = 0; i < len; i++) {
            auto & item = allowed[i];
            if (item == name) {
                return data[i];
            }
        }
    }

    template<std::size_t N>
    __attribute__((optimize("-O3"))) inline constexpr T & operator[](const char (&name)[N]) {
        return get(name);
    }

    __attribute__((optimize("-O3"))) inline constexpr T & operator[](const std::string_view name) {
        return get(name);
    }

    template<std::size_t N>
    __attribute__((optimize("-O3"))) inline constexpr const T & operator[](const char (&name)[N]) const {
        return cget(name);
    }

    __attribute__((optimize("-O3"))) inline constexpr const T & operator[](const std::string_view name) const {
        return cget(name);
    }

    inline ConstMap & operator=(const ConstMap & other) {
        for (int i = 0; i < len; i++) {
            (*this).data[i] = other.data[i];
        }
        return *this;
    }

    bool operator==(const ConstMap & other) {
        for (int i = 0; i < len; i++) {
            if ((*this).data[i] != other.data[i]) {return false;}
        }

        return true;
    }

    inline ConstMap & operator=(ConstMap && other) noexcept =default;
    ConstMap()=default;
    ConstMap(const ConstMap & o)=default;
    ConstMap(ConstMap && o) noexcept =default;
};

#endif //LIFEENGINEEXTENDED_CONSTMAP_H
