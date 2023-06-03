//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_BASE_FRONTEND_H
#define LIFEENGINEEXTENDED_BASE_FRONTEND_H

#include "common/includes/std.h"
#include "common/common_concepts.h"

namespace {
    void n(const std::string & name) {throw std::logic_error(name + " not implemented\n");}
}

namespace BaseAbstractFrontend {
    class Widget {
    public:
        virtual void show() {n("show");}
        virtual void hide() {n("hide");}
    };

    class Label: public Widget {
    public:
        template<typename T>
        requires Concepts::StringLike<T>
        void set_text(const T & item){n("set_text");}

        virtual std::string_view get_text() {n("get_text");}
    };

    class Window: public Widget {
    public:
        Window()=default;
        Window(const std::string_view & window_name,
               std::pair<int, int> starting_size,
               std::pair<int, int> starting_position){}; // Constructor should be this

        virtual void start() {n("start");};
    };
}

#endif //LIFEENGINEEXTENDED_BASE_FRONTEND_H
