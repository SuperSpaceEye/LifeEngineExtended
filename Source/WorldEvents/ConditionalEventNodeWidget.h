//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H

#include "ConditionalEventNodeWidgetUI.h"

class ConditionalEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ConditionalEventNodeWidget(QWidget * parent) {
        ui.setupUi(this);
        setParent(parent);
    }

private:
    Ui::ConditionalEventNode ui{};

private slots:
};


#endif //LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
