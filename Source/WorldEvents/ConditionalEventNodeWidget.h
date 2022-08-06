//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H

#include "ConditionalEventNodeWidgetUI.h"
#include "EventNodes.h"
#include "SimulationParametersTypes.h"

class ConditionalEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ConditionalEventNodeWidget(QWidget * parent);
    BaseEventNode * compile_node();

private:
    Ui::ConditionalEventNode ui{};

    ValueType value_type = ValueType::NONE;
    //values needed to compile node
    int update_every_n_ticks = 1;
    ConditionalMode mode = ConditionalMode::MoreOrEqual;
    int   i_value_to_compare_against = 1;
    float f_value_to_compare_against = 1;



    void init_gui();
private slots:
    void le_value_to_compare_against_slot();
    void le_update_every_n_ticks_slot();

    void cmb_condition_value_slot(QString str);
    void cmb_condition_mode_slot(QString str);
};


#endif //LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
