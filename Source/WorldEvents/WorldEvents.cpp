//
// Created by spaceeye on 01.08.22.
//

#include "WorldEvents.h"

WorldEvents::WorldEvents(Ui::MainWindow *parent_ui,
                         SimulationParameters * sp,
                         OrganismBlockParameters * bp,
                         OrganismInfoContainer * ic,
                         EngineControlParameters * ecp,
                         SimulationEngine * engine):
    parent_ui(parent_ui), pl(sp, bp, ic), ecp(ecp), engine(engine) {
    ui.setupUi(this);

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
        if (widget == nullptr || widget == NULL) {
            delete layout;
            return;
        }
        auto * spacer_item = ui.world_events_layout->itemAt(ui.world_events_layout->count()-1);
        ui.world_events_layout->removeItem(spacer_item);
        delete spacer_item;

        layout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

        widget->setMinimumSize(400, 200);
        widget->setMaximumSize(400, 200);

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
    NodeType new_node_type = NodeType::Conditional;

    QDialog main_dialog(this);

    auto change_value_button = new QPushButton("Change Value Node");
    connect(change_value_button, &QPushButton::clicked, [&new_node_type, &main_dialog](){
       new_node_type = NodeType::ChangeValue;
       main_dialog.accept();
    });

    auto conditional_button = new QPushButton("Conditional Node");
    connect(conditional_button, &QPushButton::clicked, [&new_node_type, &main_dialog](){
        new_node_type = NodeType::Conditional;
        main_dialog.accept();
    });

    QDialogButtonBox dialog(&main_dialog);
    dialog.addButton(change_value_button, QDialogButtonBox::ButtonRole::AcceptRole);
    dialog.addButton(conditional_button, QDialogButtonBox::ButtonRole::AcceptRole);

    dialog.setMinimumSize(400, 100);

    if (!main_dialog.exec()) {return nullptr;}

    QWidget * new_widget = nullptr;

    switch (new_node_type) {
        case NodeType::ChangeValue:
            new_widget = new ChangeValueEventNodeWidget(ui.world_events_widget, nullptr, pl, widget_layout, event_node_branch_starting_node_container);
            event_node_branch_starting_node_container.push_back(reinterpret_cast<ChangeValueEventNodeWidget*>(new_widget)->node);
            break;
        case NodeType::Conditional:
            new_widget = new ConditionalEventNodeWidget(ui.world_events_widget, nullptr, pl, widget_layout, event_node_branch_starting_node_container);
            event_node_branch_starting_node_container.push_back(reinterpret_cast<ConditionalEventNodeWidget*>(new_widget)->node);
            break;
    }

    new_widget->setSizePolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);

    return new_widget;
}

bool WorldEvents::verify_nodes() {
    if (event_node_branch_starting_node_container.empty()) {return false;}
    for (auto * starting_node: event_node_branch_starting_node_container) {
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
                        throw "";
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
    return true;
}

bool WorldEvents::check_conditional_node(BaseEventNode *node) {
    auto * _node = reinterpret_cast<ConditionalEventNode<double>*>(node);
    if (_node->check_value == nullptr) {return false;}
    return true;
}


