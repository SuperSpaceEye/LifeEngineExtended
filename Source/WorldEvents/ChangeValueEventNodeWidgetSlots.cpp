//
// Created by spaceeye on 06.08.22.
//

#include "ChangeValueEventNodeWidget.h"

// ==================== Line Edits ====================

void ChangeValueEventNodeWidget::le_time_horizon_slot() {
    auto _node = reinterpret_cast<ChangeValueEventNode<int32_t>*>(node);
    le_slot_lower_bound<uint32_t>(_node->time_horizon, _node->time_horizon, "int", ui.le_time_horizon, 1, "1");
}

//TODO no checks if value can be target value.
void ChangeValueEventNodeWidget::le_target_value_slot() {
    auto _node = reinterpret_cast<ChangeValueEventNode<int32_t>*>(node);
    switch (_node->value_type) {
        case ChangeTypes::INT32: {
            le_slot_no_bound(_node->target_value, _node->target_value, "int", ui.le_target_value);
        }
            break;
        case ChangeTypes::FLOAT: {
            auto _node = reinterpret_cast<ChangeValueEventNode<float>*>(node);
            le_slot_no_bound(_node->target_value, _node->target_value, "float", ui.le_target_value);
        }
            break;
    }
}

void ChangeValueEventNodeWidget::le_update_every_n_ticks_slot() {
    le_slot_lower_bound<uint32_t>(node->execute_every_n_tick, node->execute_every_n_tick, "int", ui.le_update_every_n_ticks, 1, "1");
}

// ==================== Combo Boxes ====================

void ChangeValueEventNodeWidget::cmb_change_value_slot(QString str) {
    auto name = str.toStdString();
    auto value = pl.get_changeable_value_address_from_name(name);

    switch (value.type) {
        case ValueType::NONE:
//            display_message("Choose variable");
            break;
        case ValueType::INT: {
            auto _node = reinterpret_cast<ChangeValueEventNode<int32_t> *>(node);

            if (_node->value_type == ChangeTypes::INT32) {
                _node->change_value = value.int_val;
            } else {
                auto * new_node = new ChangeValueEventNode<int32_t>(node->next_node,
                                                                    node->previous_node,
                                                                    value.int_val,
                                                                    static_cast<int32_t>(_node->target_value),
                                                                    _node->time_horizon,
                                                                    node->execute_every_n_tick,
                                                                    _node->change_mode,
                                                                    ChangeTypes::INT32);

                int index_of_node = 0;
                for (; index_of_node < starting_nodes.size(); index_of_node++) {
                    if (starting_nodes[index_of_node] == node) {
                        break;
                    }
                }
                starting_nodes[index_of_node] = new_node;

                delete node;
                node = new_node;
                if (node->previous_node != nullptr) {node->previous_node->next_node = node;}
                if (node->next_node != nullptr) {node->next_node->previous_node = node;}
            }
        }
            break;
        case ValueType::FLOAT:
            auto _node = reinterpret_cast<ChangeValueEventNode<float> *>(node);

            if (_node->value_type == ChangeTypes::FLOAT) {
                _node->change_value = value.float_val;
            } else {
                auto * new_node = new ChangeValueEventNode<float>(node->next_node,
                                                                  node->previous_node,
                                                                  value.float_val,
                                                                  static_cast<float>(_node->target_value),
                                                                  _node->time_horizon,
                                                                  node->execute_every_n_tick,
                                                                  _node->change_mode,
                                                                  ChangeTypes::FLOAT);

                int index_of_node = 0;
                for (; index_of_node < starting_nodes.size(); index_of_node++) {
                    if (starting_nodes[index_of_node] == node) {
                        break;
                    }
                }
                starting_nodes[index_of_node] = new_node;

                delete node;
                node = new_node;
                if (node->previous_node != nullptr) {node->previous_node->next_node = node;}
                if (node->next_node != nullptr) {node->next_node->previous_node = node;}
            }
            break;
    }
}

void ChangeValueEventNodeWidget::cmb_change_mode_slot(QString str) {
    auto _node = reinterpret_cast<ChangeValueEventNode<int32_t>*>(node);
    if (str == "Linear") {
        _node->change_mode = ChangeValueMode::Linear;
        ui.le_time_horizon->show();
        ui.time_horizon_label->show();
    } else if (str == "Step") {
        _node->change_mode = ChangeValueMode::Step;
        ui.le_time_horizon->hide();
        ui.time_horizon_label->hide();
    }

}

// ==================== Buttons ====================

void ChangeValueEventNodeWidget::b_new_event_left_slot() {

}

void ChangeValueEventNodeWidget::b_new_event_right_slot() {

}

void ChangeValueEventNodeWidget::b_delete_event_slot() {
    //if it's the last event in branch
    if (layout->count() == 1) {
        layout->deleteLater();
        for (int i = 0; i < starting_nodes.size(); i++) {
            if (starting_nodes[i] == node) {
                starting_nodes.erase(starting_nodes.begin() + i);
                break;
            }
        }
    }

    node->delete_node();
    this->deleteLater();
}