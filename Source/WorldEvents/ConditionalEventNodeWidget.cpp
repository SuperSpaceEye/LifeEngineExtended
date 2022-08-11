//
// Created by spaceeye on 04.08.22.
//

#include "ConditionalEventNodeWidget.h"

ConditionalEventNodeWidget::ConditionalEventNodeWidget(QWidget *parent,
                                                       BaseEventNode * previous_node,
                                                       ParametersList & parameters_list,
                                                       QHBoxLayout * layout,
                                                       std::vector<BaseEventNode*> & starting_nodes,
                                                       std::vector<char*> & repeating_branch):
    pl(parameters_list), layout(layout), starting_nodes(starting_nodes), repeating_branch(repeating_branch) {
    ui.setupUi(this);
    setParent(parent);
    this->setMinimumSize(400, 200);
    this->setMaximumSize(400, 200);

    node = new ConditionalEventNode<int64_t>(nullptr, 0, ConditionalMode::MoreOrEqual, ConditionalTypes::INT64, nullptr, previous_node,
                                             nullptr, 10);

    init_gui();
}

void ConditionalEventNodeWidget::init_gui() {
    auto parameters_list = pl.get_changing_parameters_list();
    for (auto & name: parameters_list) {
        ui.cmb_condition_value->addItem(QString::fromStdString(name));
    }

    ui.le_update_every_n_ticks->setText(QString::fromStdString(std::to_string(node->execute_every_n_tick)));
    //TODO finish else condition
    //label "else"
    ui.label_5->hide();
}

