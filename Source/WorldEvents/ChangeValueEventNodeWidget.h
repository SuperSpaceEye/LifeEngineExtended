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
    void le_time_horizon_slot();
    void le_target_value_slot();
    void le_update_every_n_ticks_slot();

    void cmb_change_value_slot(QString str);
    //should hide time_horizon_layout, time_horizon_label, le_time_horizon
    void cmb_change_mode_slot(QString str);
};


#endif //LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H
