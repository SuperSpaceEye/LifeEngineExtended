//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H

#include "ConditionalEventNodeWidgetUI.h"
#include "EventNodes.h"
#include "SimulationParametersTypes.h"
#include "../../Stuff/MiscFuncs.h"
#include "ParametersList.h"
#include "ChangeValueEventNodeWidget.h"

class ConditionalEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ConditionalEventNodeWidget(QWidget * parent,
                               BaseEventNode * previous_node,
                               ParametersList & parameters_list,
                               QHBoxLayout * layout,
                               std::vector<BaseEventNode*> & starting_nodes,
                               std::vector<char*> & repeating_branch);

    BaseEventNode * node = nullptr;
private:
    Ui::ConditionalEventNode ui{};
    ParametersList & pl;
    QHBoxLayout * layout;
    std::vector<BaseEventNode*> & starting_nodes;
    std::vector<char*> & repeating_branch;

    void init_gui();
private slots:
    void le_value_to_compare_against_slot();
    void le_update_every_n_ticks_slot();

    void cmb_condition_value_slot(QString str);
    void cmb_condition_mode_slot(QString str);

    void b_new_event_left_slot();
    void b_new_event_right_slot();
    void b_delete_event_slot();
};


#endif //LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
