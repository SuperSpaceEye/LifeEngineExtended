//
// Created by spaceeye on 06.08.22.
//

#include "ConditionalEventNodeWidget.h"

// ==================== Line Edits ====================

void ConditionalEventNodeWidget::le_value_to_compare_against_slot() {
    // we don't care if ConditionalEventNode is int64 or double because value type is 'first' element in struct.
    switch (reinterpret_cast<ConditionalEventNode<double>*>(node)->value_type) {
        case ConditionalTypes::DOUBLE: {
            auto *_node = reinterpret_cast<ConditionalEventNode<double>*>(node);
            le_slot_no_bound<double>(_node->fixed_value, _node->fixed_value, "double", ui.le_value_to_compare_against);
        }
            break;
        case ConditionalTypes::INT64: {
            auto *_node = reinterpret_cast<ConditionalEventNode<int64_t>*>(node);
            le_slot_no_bound<int64_t>(_node->fixed_value, _node->fixed_value, "int64", ui.le_value_to_compare_against);
        }
            break;
    }
}

void ConditionalEventNodeWidget::le_update_every_n_ticks_slot() {
    le_slot_lower_bound<uint32_t>(node->execute_every_n_tick, node->execute_every_n_tick, "int", ui.le_update_every_n_ticks, 1, "1");

}

// ==================== Combo boxes ====================

void ConditionalEventNodeWidget::cmb_condition_value_slot(QString str) {
    auto name = str.toStdString();
    auto value = pl.get_changing_value_address_from_name(name);

    //If the type of chosen value is the same as type of previously chosen value, then just change it.
    //If the types are different, rebuild node with new value type and reconnect it with tree.
    switch (value.type) {
        case ValueType::NONE:
            reinterpret_cast<ConditionalEventNode<double>*>(node)->value_type = ConditionalTypes::NONE;
//            display_message("Choose variable");
            return;
        case ValueType::DOUBLE: {
            auto _node = reinterpret_cast<ConditionalEventNode<double>*>(node);

            if (_node->value_type == ConditionalTypes::DOUBLE) {
                _node->check_value = value.double_val;
            } else if (_node->value_type == ConditionalTypes::INT64) {
                auto * new_node = new ConditionalEventNode<int64_t>(value.int64_val,
                                                                    static_cast<int64_t>(_node->fixed_value),
                                                                    _node->mode,
                                                                    ConditionalTypes::INT64,
                                                                    _node->next_node,
                                                                    _node->previous_node,
                                                                    _node->alternative_node,
                                                                    _node->execute_every_n_tick);

                int index_of_node = 0;
                for (; index_of_node < starting_nodes.size(); index_of_node++) {
                    if (starting_nodes[index_of_node] == node) {
                        break;
                    }
                }
                starting_nodes[index_of_node] = new_node;

                delete node;
                node = new_node;

                if (node->previous_node != nullptr)        {node->previous_node->next_node = node;}
                if (node->next_node != nullptr)            {node->next_node->previous_node = node;}
                if (new_node->alternative_node != nullptr) {new_node->alternative_node->previous_node = node;}
            }
        }
            break;
        case ValueType::INT64: {
            auto _node = reinterpret_cast<ConditionalEventNode<int64_t>*>(node);
            if (_node->value_type == ConditionalTypes::INT64) {
                _node->check_value = value.int64_val;
            } else if (_node->value_type == ConditionalTypes::DOUBLE) {
                auto * new_node = new ConditionalEventNode<double>(value.double_val,
                                                                   static_cast<double>(_node->fixed_value),
                                                                   _node->mode,
                                                                   ConditionalTypes::DOUBLE,
                                                                   _node->next_node,
                                                                   _node->previous_node,
                                                                   _node->alternative_node,
                                                                   _node->execute_every_n_tick);

                int index_of_node = 0;
                for (; index_of_node < starting_nodes.size(); index_of_node++) {
                    if (starting_nodes[index_of_node] == node) {
                        break;
                    }
                }
                starting_nodes[index_of_node] = new_node;

                delete node;
                if (node->previous_node != nullptr)        {node->previous_node->next_node = node;}
                if (node->next_node != nullptr)            {node->next_node->previous_node = node;}
                if (new_node->alternative_node != nullptr) {new_node->alternative_node->previous_node = node;}
                node = new_node;
            }
        }
            break;
    }
}

void ConditionalEventNodeWidget::cmb_condition_mode_slot(QString str) {
    if (str == "More or Equal") {
        ui.label_condition->setText(QString(">="));
        reinterpret_cast<ConditionalEventNode<double>*>(node)->mode = ConditionalMode::MoreOrEqual;
    } else if (str == "Less or Equal") {
        ui.label_condition->setText(QString("<="));
        reinterpret_cast<ConditionalEventNode<double>*>(node)->mode = ConditionalMode::LessOrEqual;
    }
}

// ==================== Buttons ====================

void ConditionalEventNodeWidget::b_new_event_left_slot() {

}

void ConditionalEventNodeWidget::b_new_event_right_slot() {

}

//TODO how tf will I delete widgets for alternative path?
void ConditionalEventNodeWidget::b_delete_event_slot() {
    //recursive_delete();
    //TODO temp workaround.
    auto _node = reinterpret_cast<ConditionalEventNode<int64_t>*>(node);
    if (_node->alternative_node != nullptr) {
        display_message("To delete this node first delete every node in alternative path.");
        return;
    }

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