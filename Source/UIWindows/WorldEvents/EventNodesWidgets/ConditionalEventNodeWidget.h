//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CONDITIONALEVENTNODEWIDGET_H

#include "UIWindows/WorldEvents/EventNodes.h"
#include "UIWindows/WorldEvents/Misc/SimulationParametersTypes.h"
#include "UIWindows/WorldEvents/Misc/ParametersList.h"

#include "Stuff/UIMisc.h"

#include "ChangeValueEventNodeWidget.h"
#include "ConditionalEventNodeWidgetUI.h"

class ConditionalEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ConditionalEventNodeWidget(QWidget * parent,
                               WorldEventNodes::BaseEventNode * previous_node,
                               ParametersList & parameters_list,
                               QHBoxLayout * layout,
                               std::vector<WorldEventNodes::BaseEventNode*> & starting_nodes,
                               std::vector<char*> & repeating_branch);

    WorldEventNodes::BaseEventNode * node = nullptr;
private:
    Ui::ConditionalEventNode ui{};
    ParametersList & pl;
    QHBoxLayout * layout;
    std::vector<WorldEventNodes::BaseEventNode*> & starting_nodes;
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
