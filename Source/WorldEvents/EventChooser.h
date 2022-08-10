//
// Created by spaceeye on 07.08.22.
//

#ifndef LIFEENGINEEXTENDED_EVENTCHOOSER_H
#define LIFEENGINEEXTENDED_EVENTCHOOSER_H

#include "EventChooserUI.h"
#include "EventNodes.h"

class EventChooser: public QWidget {
    Q_OBJECT
private:
    Ui::NewEventChooser ui;
    BaseEventNode * previous_node = nullptr;
public:
    EventChooser(QWidget * parent, BaseEventNode * previous_node);
};


#endif //LIFEENGINEEXTENDED_EVENTCHOOSER_H
