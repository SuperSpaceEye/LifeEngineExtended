//
// Created by spaceeye on 06.08.22.
//

#include "ParametersList.h"

ParametersList::ParametersList(SimulationParameters *sp, OrganismBlockParameters *bp) {
    changeable_pm["food production probability"]   = ReturnValue{ValueType::FLOAT, &sp->food_production_probability, nullptr};
    changeable_pm["lifespan multiplier"]           = ReturnValue{ValueType::FLOAT, &sp->lifespan_multiplier, nullptr};
    changeable_pm["extra reproduction cost"]       = ReturnValue{ValueType::FLOAT, &sp->extra_reproduction_cost, nullptr};
    changeable_pm["extra mover reproductive cost"] = ReturnValue{ValueType::FLOAT, &sp->extra_mover_reproductive_cost, nullptr};
    changeable_pm["global anatomy mutation rate"]  = ReturnValue{ValueType::FLOAT, &sp->global_anatomy_mutation_rate, nullptr};
    changeable_pm["global brain mutation rate"]    = ReturnValue{ValueType::FLOAT, &sp->global_brain_mutation_rate, nullptr};
    changeable_pm["killer damage amount"]          = ReturnValue{ValueType::FLOAT, &sp->killer_damage_amount, nullptr};

    changeable_pm["auto produce food every n ticks"] = ReturnValue{ValueType::INT, nullptr, &sp->auto_produce_food_every_n_ticks};
    changeable_pm["auto produce n food"]             = ReturnValue{ValueType::INT, nullptr, &sp->auto_produce_n_food};
    changeable_pm["produce food every n life ticks"] = ReturnValue{ValueType::INT, nullptr, &sp->produce_food_every_n_life_ticks};
    changeable_pm["add cell"]                        = ReturnValue{ValueType::INT, nullptr, &sp->add_cell};
    changeable_pm["change cell"]                     = ReturnValue{ValueType::INT, nullptr, &sp->change_cell};
    changeable_pm["remove cell"]                     = ReturnValue{ValueType::INT, nullptr, &sp->remove_cell};

    changeable_pm["Mouth Block food cost modifier"]= ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Mouth Block life point amount"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Mouth Block lifetime weight"]   = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Mouth Block chance weight"]     = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Producer Block food cost modifier"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Producer Block life point amount"]  = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Producer Block lifetime weight"]    = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Producer Block chance weight"]      = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Mover Block food cost modifier"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Mover Block life point amount"]  = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Mover Block lifetime weight"]    = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Mover Block chance weight"]      = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Killer Block food cost modifier"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Killer Block life point amount"]  = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Killer Block lifetime weight"]    = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Killer Block chance weight"]      = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Armor Block food cost modifier"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Armor Block life point amount"]  = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Armor Block lifetime weight"]    = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Armor Block chance weight"]      = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Eye Block food cost modifier"] = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Eye Block life point amount"]  = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Eye Block lifetime weight"]    = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Eye Block chance weight"]      = ReturnValue{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};
}

std::vector<std::pair<std::string, std::string>> ParametersList::get_changeable_parameters_list() {
    std::vector<std::pair<std::string, std::string>> return_vector{};

    for (auto & [key, value]: changeable_pm) {
        std::string type_str;
        switch (value.type) {
            case ValueType::INT:
                type_str = "int";
                break;
            case ValueType::FLOAT:
                type_str = "float";
                break;
            default:
                throw "";
        }

        return_vector.emplace_back(std::pair<std::string, std::string>(key, type_str));
    }

    return return_vector;
}

std::vector<std::pair<std::string, std::string>> ParametersList::get_changing_parameters_list() {
    std::vector<std::pair<std::string, std::string>> return_vector{};

    for (auto & [key, value]: changing_pm) {
        std::string type_str;
        switch (value.type) {
            case ValueType::INT:
                type_str = "int";
                break;
            case ValueType::FLOAT:
                type_str = "float";
                break;
            default:
                throw "";
        }

        return_vector.emplace_back(std::pair<std::string, std::string>(key, type_str));
    }

    return return_vector;
};

ReturnValue ParametersList::get_changeable_value_address_from_name(std::string & name) {
    return changeable_pm[name];
}

ReturnValue ParametersList::get_changing_value_address_from_name(std::string & name) {
    return changing_pm[name];
}
