//
// Created by spaceeye on 17.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCPARAMETERS_H
#define LIFEENGINEEXTENDED_OCCPARAMETERS_H

#include "OCCParametersUI.h"
#include "../MainWindow/WindowUI.h"

class OCCParametersWindow: public QWidget {
    Q_OBJECT
private:
    Ui::OCCParametes ui{};
    Ui::MainWindow * parent_ui = nullptr;

public:
    OCCParametersWindow(Ui::MainWindow * parent_ui);
};


#endif //LIFEENGINEEXTENDED_OCCPARAMETERS_H
