//
// Created by spaceeye on 04.06.23.
//

#ifndef LIFEENGINEEXTENDED_MAINWINDOW_H
#define LIFEENGINEEXTENDED_MAINWINDOW_H

#include "Frontend/frontend_interface.h"

class MainWindow: public UiFrontend::UiWindow {
public:
    MainWindow(const std::string & name,
               const std::pair<int, int> & starting_size,
               const std::pair<int, int> & starting_pos);
};


#endif //LIFEENGINEEXTENDED_MAINWINDOW_H
