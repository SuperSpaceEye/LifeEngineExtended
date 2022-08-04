//
// Created by spaceeye on 01.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTS_H
#define LIFEENGINEEXTENDED_WORLDEVENTS_H

#include "WorldEventsUI.h"
#include "../MainWindow/WindowUI.h"
#include "EventNodes.h"

class WorldEvents: public QWidget {
    Q_OBJECT
private:
    Ui::WorldEvents ui{};
    Ui::MainWindow * parent_ui = nullptr;

    std::vector<BaseEventNode*> event_node_storage;

    void closeEvent(QCloseEvent * event) override;
public:
    WorldEvents(Ui::MainWindow * parent_ui);
};


#endif //LIFEENGINEEXTENDED_WORLDEVENTS_H
