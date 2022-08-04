//
// Created by spaceeye on 04.08.22.
//

#ifndef LIFEENGINEEXTENDED_EVENTNODES_H
#define LIFEENGINEEXTENDED_EVENTNODES_H

enum class NodeType {
    ChangeValue,
    Conditional,
};

//

struct BaseEventNode {
    NodeType type;
    BaseEventNode * next_node = nullptr;
    uint32_t execute_every_n_tick = 1;

    BaseEventNode()=delete;

    BaseEventNode * update(uint32_t current_time) {
        if (current_time - last_execution_time >= execute_every_n_tick) {
            last_execution_time = execute_every_n_tick;
            return _update(current_time);
        }
        return this;
    }

private:
    uint32_t last_execution_time = 0;
    virtual BaseEventNode * _update(uint32_t current_time)=0;
};

enum class ChangeValueMode {
    Step,
    Linear,
//    Exponential
//    Logarithmic
};

//Will update the selected value every time update() is called. When it's finished updating, will return pointer to the next node, otherwise to itself.
template <typename T>
struct ChangeValueEventNode: public BaseEventNode {
    T * change_value;
    T target_value;
    T start_value;
    ChangeValueMode change_mode;
    uint32_t time_horizon;
    uint32_t last_updated = 0;
    uint32_t total_updated = 0;
    bool execution_started = false;
    ChangeValueEventNode(BaseEventNode * _next_node,
                         T * value_to_change,
                         T _target_value,
                         uint32_t _time_horizon,
                         uint32_t _execute_every_n_tick,
                         ChangeValueMode mode) {
        type = NodeType::ChangeValue;
        next_node = _next_node;
        change_value = value_to_change;
        target_value = _target_value;
        time_horizon = _time_horizon;
        execute_every_n_tick = _execute_every_n_tick;
        change_mode = mode;
    }

    BaseEventNode * _update(uint32_t current_time) override {
        if (!execution_started) {last_updated = current_time; start_value = *change_value;}

        if (change_mode == ChangeValueMode::Step) {
            *change_value = target_value;
            return next_node;
        }

        auto time_difference = current_time - last_updated;
        total_updated += time_difference;
        last_updated = current_time;

        if (time_horizon - total_updated <= 0) {
            *change_value = target_value;
            reset_node();
            return next_node;
        }

        switch (change_mode) {
            case ChangeValueMode::Linear:
                *change_value = (target_value - start_value) * total_updated / time_horizon + start_value;
                break;
        }
        return this;
    }

    void reset_node() {
        execution_started = false;
        total_updated = 0;
    }
};

enum class ConditionalMode {
    MoreOrEqual,
    LessOrEqual,
};

//If no alternative node, will repeatedly check the condition, and only return next node if it is true.
//If there is an alternative node, will return next node if the condition is true, and alternative node if false.

//(Not for now.) If another_value is nullptr, will check against fixed, else will check against another value
template<typename T>
struct ConditionalEventNode: public BaseEventNode {
    BaseEventNode * alternative_node = nullptr;
    T * check_value;
//    T * another_value;
    T fixed_value;
    ConditionalMode mode;

//    bool check_against_fixed = false;

//    ConditionalEventNode(T * check_value,T * another_value,T fixed_value):
//    check_value(check_value),another_value(another_value),fixed_value(fixed_value) {
//        if (another_value == nullptr) {check_against_fixed = true;}
//        type = NodeType::Conditional;
//    }

    ConditionalEventNode(T * check_value, T fixed_value, ConditionalMode mode):
            check_value(check_value), fixed_value(fixed_value), mode(mode) {
        type = NodeType::Conditional;
    }

    BaseEventNode * _update(uint32_t current_time) override {
        bool condition_true = false;
//        if (check_against_fixed) {
//            if (*check_value >= fixed_value   ) {condition_true = true;}
//        } else {
//            if (*check_value >= *another_value) {condition_true = true;}
//        }

        switch (mode) {
            case ConditionalMode::MoreOrEqual:
                if (*check_value >= fixed_value) {condition_true = true;}
                break;
            case ConditionalMode::LessOrEqual:
                if (*check_value <= fixed_value) {condition_true = true;}
                break;
        }

        if (condition_true) {
            return next_node;
        } else {
            if (alternative_node != nullptr) {
                return alternative_node;
            }
        }
        return this;
    }
};

#endif //LIFEENGINEEXTENDED_EVENTNODES_H
