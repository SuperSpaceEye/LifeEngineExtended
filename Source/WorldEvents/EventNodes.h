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

template <typename T>
struct ConditionalEventNode;

struct BaseEventNode {
    NodeType type;
    BaseEventNode * next_node = nullptr;
    BaseEventNode * previous_node = nullptr;
    BaseEventNode * alternative_node = nullptr;
    uint32_t execute_every_n_tick = 1;
    bool alternative_from_conditional = false;

    BaseEventNode()= default;

    BaseEventNode * update(uint32_t current_time) {
        if (current_time - last_execution_time >= execute_every_n_tick) {
            last_execution_time = execute_every_n_tick;
            return _update(current_time);
        }
        return this;
    }

    void delete_node() {
        if (previous_node != nullptr && !alternative_from_conditional) {
            previous_node->next_node = next_node;
        } else if (previous_node != nullptr) {
            previous_node->alternative_node = next_node;
        }
        if (next_node != nullptr) {next_node->previous_node = previous_node;}

        delete this;
    }

    void delete_from_this() {
        if (next_node != nullptr) {
            next_node->delete_from_this();
        }
        if (alternative_node != nullptr) {
            alternative_node->delete_from_this();
        }

        previous_node->next_node = nullptr;
        delete this;
    }

    virtual ~BaseEventNode()=default;

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

enum class ChangeTypes {
    INT32,
    FLOAT
};

//Will update the selected value every time update() is called. When it's finished updating, will return pointer to the next node, otherwise to itself.
template <typename T>
struct ChangeValueEventNode: public BaseEventNode {
    ChangeTypes value_type;
    ChangeValueMode change_mode;
    uint32_t time_horizon = 1;
    uint32_t last_updated = 0;
    uint32_t total_updated = 0;
    bool execution_started = false;
    T * change_value;
    T target_value;
    T start_value;
    ChangeValueEventNode(BaseEventNode * _next_node,
                         BaseEventNode * _previous_node,
                         T * value_to_change,
                         T target_value,
                         uint32_t time_horizon,
                         uint32_t _execute_every_n_tick,
                         ChangeValueMode mode,
                         ChangeTypes value_type):
                         value_type(value_type),
                         change_mode(mode),
                         time_horizon(time_horizon),
                         target_value(target_value),
                         change_value(value_to_change)
                         {
        type = NodeType::ChangeValue;
        next_node = _next_node;
        previous_node = _previous_node;
        execute_every_n_tick = _execute_every_n_tick;
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

    ~ChangeValueEventNode() override =default;
};

enum class ConditionalMode {
    MoreOrEqual,
    LessOrEqual,
};

enum class ConditionalTypes {
    DOUBLE,
    INT64,
};

//If no alternative node, will repeatedly check the condition, and only return next node if it is true.
//If there is an alternative node, will return next node if the condition is true, and alternative node if false.

//(Not for now.) If another_value is nullptr, will check against fixed, else will check against another value
template<typename T>
struct ConditionalEventNode: public BaseEventNode {
    ConditionalTypes value_type;
    ConditionalMode mode;
    T * check_value;
//    T * another_value;
    T fixed_value;

//    bool check_against_fixed = false;

//    ConditionalEventNode(T * check_value,T * another_value,T fixed_value):
//    check_value(check_value),another_value(another_value),fixed_value(fixed_value) {
//        if (another_value == nullptr) {check_against_fixed = true;}
//        type = NodeType::Conditional;
//    }

    ConditionalEventNode(T * check_value, T fixed_value, ConditionalMode mode, ConditionalTypes value_type, BaseEventNode * _next_node,
                         BaseEventNode * _previous_node, BaseEventNode * _alternative_node, uint32_t _execute_every_n_tick):
            check_value(check_value), fixed_value(fixed_value), mode(mode), value_type(value_type) {
        type = NodeType::Conditional;
        next_node = _next_node;
        previous_node = _previous_node;
        alternative_node = _alternative_node;
        execute_every_n_tick = _execute_every_n_tick;
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

    void delete_node() {
        if (previous_node != nullptr) {previous_node->next_node = next_node;}
        if (next_node != nullptr)     {next_node->previous_node = previous_node;}
        if (alternative_node != nullptr) {alternative_node->delete_from_this();}

        delete this;
    }

    ~ConditionalEventNode() override =default;
};

#endif //LIFEENGINEEXTENDED_EVENTNODES_H
