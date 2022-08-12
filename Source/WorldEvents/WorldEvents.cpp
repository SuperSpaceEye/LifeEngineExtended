//
// Created by spaceeye on 01.08.22.
//

#include "WorldEvents.h"
#include "WorldEventsEnums.h"

WorldEvents::WorldEvents(Ui::MainWindow *parent_ui,
                         SimulationParameters * sp,
                         OrganismBlockParameters * bp,
                         OrganismInfoContainer * ic,
                         EngineControlParameters * ecp,
                         SimulationEngine * engine):
    parent_ui(parent_ui), pl(sp, bp, ic), ecp(ecp), engine(engine) {
    ui.setupUi(this);

    ui.le_update_world_events_every_n->setText(QString::fromStdString(std::to_string(ecp->update_world_events_every_n_tick)));
    ui.le_collect_info_every_n->setText(QString::fromStdString(std::to_string(ecp->update_info_every_n_tick)));

    ui.world_events_layout->setAlignment(Qt::AlignLeft);

    button_add_new_branch();
}

void WorldEvents::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_info_window->setChecked(false);
    QWidget::closeEvent(event);
}

void WorldEvents::button_add_new_branch() {
    auto * button = new QPushButton("Add new event branch", this);
    button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    connect(button, &QPushButton::clicked, [&, button](){
        auto * layout = new QHBoxLayout{};
        auto * widget = node_chooser(layout);
        if (widget == nullptr) {
            delete layout;
            return;
        }
        auto * spacer_item = ui.world_events_layout->itemAt(ui.world_events_layout->count()-1);
        ui.world_events_layout->removeItem(spacer_item);
        delete spacer_item;

        auto repeating_branch_pointer = new char(true);

        repeating_branch.emplace_back(repeating_branch_pointer);

        auto * check_box = new QCheckBox("Repeat branch", this);
        check_box->setChecked(true);
        connect(check_box, &QCheckBox::clicked, [&, repeating_branch_pointer](){
            *repeating_branch_pointer = !*repeating_branch_pointer;
        });

        layout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

        layout->addWidget(check_box);
        layout->addWidget(widget);

        ui.world_events_layout->addLayout(layout);

        button_add_new_branch();
        button->deleteLater();
    });

    auto * v_spacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);

    ui.world_events_layout->addWidget(button);
    ui.world_events_layout->addItem(v_spacer);
}

QWidget * WorldEvents::node_chooser(QHBoxLayout * widget_layout) {
    NodeType new_node_type;

    if (!choose_node_window(new_node_type)) {return nullptr;}

    QWidget * new_widget = nullptr;

    switch (new_node_type) {
        case NodeType::ChangeValue:
            new_widget = new ChangeValueEventNodeWidget(ui.world_events_widget, nullptr, pl, widget_layout, starting_nodes_container, repeating_branch);
            starting_nodes_container.push_back(reinterpret_cast<ChangeValueEventNodeWidget*>(new_widget)->node);
            break;
        case NodeType::Conditional:
            new_widget = new ConditionalEventNodeWidget(ui.world_events_widget, nullptr, pl, widget_layout, starting_nodes_container, repeating_branch);
            starting_nodes_container.push_back(reinterpret_cast<ConditionalEventNodeWidget*>(new_widget)->node);
            break;
    }

    new_widget->setSizePolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);

    return new_widget;
}

bool WorldEvents::verify_nodes() {
    if (starting_nodes_container.empty()) {return false;}
    for (auto * starting_node: starting_nodes_container) {
        std::vector<BaseEventNode*> branch_stack;
        branch_stack.push_back(starting_node);
        while (!branch_stack.empty()) {
            auto node = branch_stack.back();
            branch_stack.pop_back();

            while (node != nullptr) {
                switch (node->type) {
                    case NodeType::ChangeValue:
                        if (!check_change_value_node(node)) { return false; }
                        break;
                    case NodeType::Conditional:
                        if (!check_conditional_node(node)) { return false; }
                        if (node->alternative_node != nullptr) {branch_stack.push_back(node->alternative_node);}
                        break;
                    default:
                        return false;
                }
                node = node->next_node;
            }
        }
    }
    return true;
}

bool WorldEvents::check_change_value_node(BaseEventNode *node) {
    auto * _node = reinterpret_cast<ChangeValueEventNode<float>*>(node);
    if (_node->change_value == nullptr) {return false;}
    if (_node->value_type   == ChangeTypes::NONE) {return false;}
    return true;
}

bool WorldEvents::check_conditional_node(BaseEventNode *node) {
    auto * _node = reinterpret_cast<ConditionalEventNode<double>*>(node);
    if (_node->check_value == nullptr) {return false;}
    if (_node->value_type  == ConditionalTypes::NONE) {return false;}
    return true;
}

BaseEventNode * WorldEvents::copy_node(BaseEventNode *node, std::vector<BaseEventNode *> &node_storage) {
    if (node == nullptr) {return nullptr;}
    BaseEventNode * new_node;

    switch (node->type) {
        case NodeType::ChangeValue: {
            switch (reinterpret_cast<ChangeValueEventNode<float> *>(node)->value_type) {
                case ChangeTypes::INT32: {
                    auto * _node = reinterpret_cast<ChangeValueEventNode<int32_t> *>(node);
                    new_node = new ChangeValueEventNode<int32_t>(nullptr,
                                                                 nullptr,
                                                                 _node->change_value,
                                                                 _node->target_value,
                                                                 _node->time_horizon,
                                                                 _node->execute_every_n_tick,
                                                                 _node->change_mode,
                                                                 _node->value_type,
                                                                 _node->clamp_mode,
                                                                 _node->min_clamp_value,
                                                                 _node->max_clamp_value);
                }
                    break;
                case ChangeTypes::FLOAT: {
                    auto * _node = reinterpret_cast<ChangeValueEventNode<float> *>(node);
                    new_node = new ChangeValueEventNode<float>(nullptr,
                                                               nullptr,
                                                               _node->change_value,
                                                               _node->target_value,
                                                               _node->time_horizon,
                                                               _node->execute_every_n_tick,
                                                               _node->change_mode,
                                                               _node->value_type,
                                                               _node->clamp_mode,
                                                               _node->min_clamp_value,
                                                               _node->max_clamp_value);
                }
                    break;
            }
        }
            break;
        case NodeType::Conditional: {
            switch (reinterpret_cast<ConditionalEventNode<double> *>(node)->value_type) {
                case ConditionalTypes::DOUBLE: {
                    auto * _node = reinterpret_cast<ConditionalEventNode<double>*>(node);
                    new_node = new ConditionalEventNode<double>(_node->check_value,
                                                                _node->fixed_value,
                                                                _node->mode,
                                                                _node->value_type,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                _node->execute_every_n_tick);
                }
                    break;
                case ConditionalTypes::INT64: {
                    auto * _node = reinterpret_cast<ConditionalEventNode<int64_t>*>(node);
                    new_node = new ConditionalEventNode<int64_t>(_node->check_value,
                                                                 _node->fixed_value,
                                                                 _node->mode,
                                                                 _node->value_type,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 _node->execute_every_n_tick);
                }
                    break;
            }
        }
            break;
    }

    new_node->next_node = copy_node(node->next_node, node_storage);
    new_node->alternative_node = copy_node(node->alternative_node, node_storage);

    node_storage.emplace_back(new_node);

    return new_node;
}

void WorldEvents::copy_nodes(std::vector<BaseEventNode *> &start_nodes, std::vector<BaseEventNode *> &node_storage) {
    for (auto * starting_node: starting_nodes_container) {
        auto * copied_root_node = copy_node(starting_node, node_storage);
        start_nodes.emplace_back(copied_root_node);
    }

    //TODO make reserve
    start_nodes.shrink_to_fit();
    node_storage.shrink_to_fit();
}
