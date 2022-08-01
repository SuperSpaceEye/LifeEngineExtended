//
// Created by spaceeye on 01.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTS_H
#define LIFEENGINEEXTENDED_WORLDEVENTS_H

#include "WorldEventsUI.h"
#include "../MainWindow/WindowUI.h"

class WorldEvents: public QWidget {
    Q_OBJECT
private:
    Ui::WorldEvents _ui{};
    Ui::MainWindow * parent_ui = nullptr;

    void closeEvent(QCloseEvent * event) override;
public:
    WorldEvents(Ui::MainWindow * parent_ui);
};


#endif //LIFEENGINEEXTENDED_WORLDEVENTS_H
