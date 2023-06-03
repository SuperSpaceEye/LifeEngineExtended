//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_QTWINDOW_H
#define LIFEENGINEEXTENDED_QTWINDOW_H

#include "common/includes/qt.h"
#include "Frontend/base_frontend.h"

namespace FrontendImpl {
    class Window: public BaseAbstractFrontend::Window, private QWindow {
    private:

    public:
        Window(const std::string_view & window_name,
               std::pair<int, int> starting_size,
               std::pair<int, int> starting_position) {

        }
    };
}

#endif //LIFEENGINEEXTENDED_QTWINDOW_H
