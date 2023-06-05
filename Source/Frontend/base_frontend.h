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

    class Label {
    public:
        template<typename T>
        requires Concepts::StringLike<T>
        void set_text(const T & item){n("set_text");}

        virtual std::string get_text() const {n("get_text");}
    };

    class Window {
    public:
        Window()=default;
        Window(const std::string & window_name,
               const std::pair<int, int> & starting_size,
               const std::pair<int, int> & starting_position){}; // Constructor should be this

        virtual void set_window_name(const std::string & name){n("set_window_name");};
        virtual std::string get_window_name() const {n("get_window_name");}
        virtual void set_focus() {n("set_focus");}
        virtual void set_size(const std::pair<int, int> & size) {n("set_size");}
        virtual std::pair<int, int> get_size() const {n("get_size");}
        virtual void set_pos(const std::pair<int, int> & pos) {n("set_pos");}
        virtual std::pair<int, int> get_pos() const {n("get_pos");}
    };
}

#endif //LIFEENGINEEXTENDED_BASE_FRONTEND_H
