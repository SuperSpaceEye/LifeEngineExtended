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
        virtual void show_it() {n("show_it");}
        virtual void hide_it() {n("hide_it");}
    };

    class Label: public Widget {
    public:
        template<typename T>
        requires Concepts::StringLike<T>
        void set_text(const T & item){n("set_text");}

        virtual std::string get_text() const {n("get_text");}
    };

    class Window: public Widget {
    public:
        Window()=default;
        Window(const std::string & window_name,
               const std::pair<int, int> & starting_size,
               const std::pair<int, int> & starting_position){}; // Constructor should be this

        virtual void set_window_name(const std::string & name){n("set_window_name");};
        virtual std::string get_window_name() const {n("get_window_name");}
    };
}

#endif //LIFEENGINEEXTENDED_BASE_FRONTEND_H
