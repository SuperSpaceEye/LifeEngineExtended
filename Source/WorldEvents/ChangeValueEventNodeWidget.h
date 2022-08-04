//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H

#include "ChangeValueEventNodeWidgetUI.h"

class ChangeValueEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ChangeValueEventNodeWidget(QWidget * parent) {
        ui.setupUi(this);
        setParent(parent);
    }

private:
    Ui::ChangeValueEventNodeWidget ui{};

private slots:

};


#endif //LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H
