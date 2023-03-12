//
// Created by spaceeye on 04.08.22.
//

#include "UIWindows/WorldEvents/Misc/WorldEventsEnums.h"

#include "ChangeValueEventNodeWidget.h"

ChangeValueEventNodeWidget::ChangeValueEventNodeWidget(QWidget * parent,
                                                       BaseEventNode * previous_node,
                                                       ParametersList & parameter_list,
                                                       QHBoxLayout * layout,
                                                       std::vector<BaseEventNode*> & starting_nodes,
                                                       std::vector<char*> & repeating_branch):
    pl(parameter_list), layout(layout), starting_nodes(starting_nodes), repeating_branch(repeating_branch) {
    ui.setupUi(this);
    setParent(parent);
    this->setMinimumSize(400, 200);
    this->setMaximumSize(400, 200);

    node = new ChangeValueEventNode<int32_t>(nullptr, previous_node, nullptr, 0, 20, 1, ChangeValueMode::Linear,
                                             ChangeTypes::INT32, ClampModes::NoClamp, 0, 0);
    init_gui();
}

void ChangeValueEventNodeWidget::init_gui() {
    auto _node = reinterpret_cast<ChangeValueEventNode<int32_t>*>(node);
    auto parameters_list = pl.get_changeable_parameters_list();
    for (auto & name: parameters_list) {
        ui.cmb_change_value->addItem(QString::fromStdString(name));
    }

    ui.le_update_every_n_ticks->setText(QString::fromStdString(std::to_string(node->execute_every_n_tick)));
//    ui.cmb_change_value->setCurrentText("None");
    //TODO for some reason the value of time horizon is mangled when I try to set default number in le, so I just won't
    // set the value at the start.
//    ui.le_time_horizon->setText(QString::fromStdString(std::to_string(_node->time_horizon)));

    if (_node->value_type == ChangeTypes::INT32) {
        ui.le_target_value->setText(QString::fromStdString(std::to_string(_node->target_value)));
    } else if (_node->value_type == ChangeTypes::FLOAT) {
        auto _node = reinterpret_cast<ChangeValueEventNode<float>*>(node);
        ui.le_target_value->setText(QString::fromStdString(std::to_string(_node->target_value)));
    }
}