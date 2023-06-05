//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_QTWINDOW_H
#define LIFEENGINEEXTENDED_QTWINDOW_H

#include "common/includes/qt.h"
#include "Frontend/qt/impl/QtWidget.h"
#include "Frontend/base_frontend.h"

namespace FrontendImpl {
    class Window: public BaseAbstractFrontend::Window, public FrontendImpl::Widget {
    public:
        void set_window_name(const std::string &name) override {base->setWindowTitle(QString::fromStdString(name));}
        std::string get_window_name() const override {return base->windowTitle().toStdString();}
        void set_focus() override {base->setFocus();}
        void set_size(const std::pair<int, int> &size) override {base->resize(size.first, size.second);}
        std::pair<int, int> get_size() const override {const auto qsize = base->size(); return {qsize.width(), qsize.height()};}
        void set_pos(const std::pair<int, int> &pos) override {base->move(base->mapToGlobal(QPoint(pos.first, pos.second)));}

        Window(const std::string & window_name,
               const std::pair<int, int> & starting_size,
               const std::pair<int, int> & starting_position) {
            base = new QWidget();
            set_window_name(window_name);
            set_size(starting_size);
            set_pos(starting_position);
        }

        ~Window() {delete base;}
    };
}

#endif //LIFEENGINEEXTENDED_QTWINDOW_H
