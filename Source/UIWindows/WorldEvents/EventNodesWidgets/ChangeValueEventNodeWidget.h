//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H
#define LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H

#include "UIWindows/WorldEvents/EventNodes.h"
#include "UIWindows/WorldEvents/Misc/ParametersList.h"

#include "Stuff/UIMisc.h"

#include "ChangeValueEventNodeWidgetUI.h"
#include "ConditionalEventNodeWidget.h"

class ChangeValueEventNodeWidget: public QWidget {
    Q_OBJECT
public:
    ChangeValueEventNodeWidget(QWidget * parent,
                               WorldEventNodes::BaseEventNode * previous_node,
                               ParametersList & pl, QHBoxLayout * layout,
                               std::vector<WorldEventNodes::BaseEventNode*> & starting_nodes,
                               std::vector<char*> & repeating_branch);
    WorldEventNodes::BaseEventNode * node = nullptr;
private:
    Ui::ChangeValueEventNodeWidget ui{};
    ParametersList & pl;
    QHBoxLayout * layout;
    std::vector<WorldEventNodes::BaseEventNode*> & starting_nodes;
    std::vector<char*> & repeating_branch;

    void init_gui();

private slots:
    void le_time_horizon_slot();
    void le_target_value_slot();
    void le_update_every_n_ticks_slot();

    void cmb_change_value_slot(QString str);
    //should hide time_horizon_layout, time_horizon_label, le_time_horizon
    void cmb_change_mode_slot(QString str);

    void b_new_event_left_slot();
    void b_new_event_right_slot();
    void b_delete_event_slot();
};


#endif //LIFEENGINEEXTENDED_CHANGEVALUEEVENTNODEWIDGET_H
