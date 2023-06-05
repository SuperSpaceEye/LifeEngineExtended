//
// Created by spaceeye on 05.06.23.
//

#ifndef LIFEENGINEEXTENDED_QTWIDGET_H
#define LIFEENGINEEXTENDED_QTWIDGET_H

#include "common/includes/qt.h"
#include "Frontend/base_frontend.h"

namespace FrontendImpl {
    class Widget: public BaseAbstractFrontend::Widget {
    protected:
        QWidget * base = nullptr;
    public:
        void show() override {base->show();}
        void hide() override {base->hide();}
        QWidget * get_base() {return base;}
    };
}

#endif //LIFEENGINEEXTENDED_QTWIDGET_H
