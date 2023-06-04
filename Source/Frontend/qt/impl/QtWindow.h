//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_QTWINDOW_H
#define LIFEENGINEEXTENDED_QTWINDOW_H

#include "common/includes/qt.h"
#include "Frontend/base_frontend.h"

namespace FrontendImpl {
    class Window: private QWidget, public BaseAbstractFrontend::Window {
    private:

    public:
        void show_it() override {this->show();}
        void hide_it() override {this->hide();}
        void set_window_name(const std::string &name) override {this->setWindowTitle(QString::fromStdString(name));}
        std::string get_window_name() const override {return this->windowTitle().toStdString();}

        Window(const std::string & window_name,
               const std::pair<int, int> & starting_size,
               const std::pair<int, int> & starting_position) {
            set_window_name(window_name);
        }
    };
}

#endif //LIFEENGINEEXTENDED_QTWINDOW_H
