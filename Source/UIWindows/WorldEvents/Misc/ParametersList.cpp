//
// Created by spaceeye on 06.08.22.
//

#include "ParametersList.h"

ParametersList::ParametersList(SimulationParameters *sp, OrganismBlockParameters *bp, OrganismInfoContainer * ic):
    sp(sp), bp(bp), ic(ic) {
    changeable_pm_list = std::vector<std::string>{
        "None",
        "==========",
        "food production probability",
        "lifespan multiplier",
        "extra reproduction cost",
        "extra mover reproductive cost",
        "global anatomy mutation rate",
        "global brain mutation rate",
        "killer damage amount",
        "anatomy mutation rate delimiter",
        "brain mutation rate delimiter",
        "auto produce food every n ticks",
        "auto produce n food",
        "produce food every n life ticks",
        "==========",
        "add cell",
        "change cell",
        "remove cell",
    };

    changeable_pm_list.reserve(changeable_pm_list.size() + NUM_ORGANISM_BLOCKS*5);

    for (const auto & name: ORGANISM_BLOCK_NAMES) {
        changeable_pm_list.emplace_back("==========");
        changeable_pm_list.emplace_back(name+" Block food cost modifier");
        changeable_pm_list.emplace_back(name+" Block life point amount");
        changeable_pm_list.emplace_back(name+" Block food lifetime weight");
        changeable_pm_list.emplace_back(name+" Block food chance weight");
    }


    changeable_pm["None"] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};
    changeable_pm["=========="] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};

    changeable_pm["food production probability"]   = ChangeableReturn{ValueType::FLOAT, &sp->food_production_probability,   nullptr, ClampModes::ClampMinMaxValues, 0, 1};
    changeable_pm["lifespan multiplier"]           = ChangeableReturn{ValueType::FLOAT, &sp->lifespan_multiplier,           nullptr, ClampModes::ClampMinValue,     0};
    changeable_pm["extra reproduction cost"]       = ChangeableReturn{ValueType::FLOAT, &sp->extra_reproduction_cost,       nullptr};
    changeable_pm["extra mover reproductive cost"] = ChangeableReturn{ValueType::FLOAT, &sp->extra_mover_reproductive_cost, nullptr};
    changeable_pm["global anatomy mutation rate"]  = ChangeableReturn{ValueType::FLOAT, &sp->global_anatomy_mutation_rate,  nullptr, ClampModes::ClampMinMaxValues, 0, 1};
    changeable_pm["global brain mutation rate"]    = ChangeableReturn{ValueType::FLOAT, &sp->global_brain_mutation_rate,    nullptr, ClampModes::ClampMinMaxValues, 0, 1};
    changeable_pm["killer damage amount"]          = ChangeableReturn{ValueType::FLOAT, &sp->killer_damage_amount,          nullptr, ClampModes::ClampMinValue,     0};

    changeable_pm["anatomy mutation rate delimiter"] = ChangeableReturn{ValueType::FLOAT, &sp->anatomy_mutation_rate_delimiter, nullptr, ClampModes::ClampMinMaxValues, 0, 1};
    changeable_pm["brain mutation rate delimiter"]   = ChangeableReturn{ValueType::FLOAT, &sp->brain_mutation_rate_delimiter,   nullptr, ClampModes::ClampMinMaxValues, 0, 1};

    changeable_pm["auto produce food every n ticks"] = ChangeableReturn{ValueType::INT, nullptr, &sp->auto_produce_food_every_n_ticks, ClampModes::ClampMinValue, 0, 0, 0};
    changeable_pm["auto produce n food"]             = ChangeableReturn{ValueType::INT, nullptr, &sp->auto_produce_n_food,             ClampModes::ClampMinValue, 0, 0, 0};
    changeable_pm["produce food every n life ticks"] = ChangeableReturn{ValueType::INT, nullptr, &sp->produce_food_every_n_life_ticks, ClampModes::ClampMinValue, 0, 0, 1};
    changeable_pm["add cell"]                        = ChangeableReturn{ValueType::INT, nullptr, &sp->add_cell,                        ClampModes::ClampMinValue, 0, 0, 0};
    changeable_pm["change cell"]                     = ChangeableReturn{ValueType::INT, nullptr, &sp->change_cell,                     ClampModes::ClampMinValue, 0, 0, 0};
    changeable_pm["remove cell"]                     = ChangeableReturn{ValueType::INT, nullptr, &sp->remove_cell,                     ClampModes::ClampMinValue, 0, 0, 0};

    for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
        const auto & block = ORGANISM_BLOCK_NAMES[i];
        changeable_pm[block+" Block food cost modifier"]= ChangeableReturn{ValueType::FLOAT, &bp->pa[i].food_cost,         nullptr};
        changeable_pm[block+" Block life point amount"] = ChangeableReturn{ValueType::FLOAT, &bp->pa[i].life_point_amount, nullptr};
        changeable_pm[block+" Block lifetime weight"]   = ChangeableReturn{ValueType::FLOAT, &bp->pa[i].lifetime_weight,   nullptr};
        changeable_pm[block+" Block chance weight"]     = ChangeableReturn{ValueType::FLOAT, &bp->pa[i].chance_weight,     nullptr, ClampModes::ClampMinValue, 0};
    }

    changing_pm_list = std::vector<std::string> {
            "None",
            "==========",
            "avg total organisms size",
            "avg total organisms lifetime",
            "avg total organisms age",
    };

    auto lower = [](std::string name){std::string data;for (auto & chr: name) {data += tolower(chr);}return data;};

    for (auto & name: ORGANISM_BLOCK_NAMES) {
        changing_pm_list.emplace_back("avg total organism "+lower(name)+" blocks");
    }

    auto temp = std::vector<std::string> {
            "avg total organisms brain mutation rate",
            "avg total organisms anatomy mutation rate",
            "total organisms amount",
            "==========",
            "avg stationary organisms size",
            "avg stationary organisms lifetime",
            "avg stationary organisms age",
    };
    changing_pm_list.insert(changing_pm_list.end(), temp.begin(), temp.end());

    for (auto & name: ORGANISM_BLOCK_NAMES) {
        changing_pm_list.emplace_back("avg stationary organism "+lower(name)+" blocks");
    }

    temp = std::vector<std::string> {
            "avg stationary organisms brain mutation rate",
            "avg stationary organisms anatomy mutation rate",
            "stationary organisms amount",
            "==========",
            "moving organisms with eyes",
            "avg moving organisms size",
            "avg moving organisms lifetime",
            "avg moving organisms age",
    };
    changing_pm_list.insert(changing_pm_list.end(), temp.begin(), temp.end());

    for (auto & name: ORGANISM_BLOCK_NAMES) {
        changing_pm_list.emplace_back("avg moving organism "+lower(name)+" blocks");
    }

    temp = std::vector<std::string> {
            "avg moving organisms brain mutation rate",
            "avg moving organisms anatomy mutation rate",
            "total organisms amount",
    };
    changing_pm_list.insert(changing_pm_list.end(), temp.begin(), temp.end());

    changing_pm["None"] = ChangingReturn{ValueType::NONE, nullptr, nullptr};
    changeable_pm["=========="] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};

    changing_pm["moving organisms with eyes"] = ChangingReturn{ValueType::INT64, nullptr, &ic->organisms_with_eyes};

    std::array<std::string, 3> names {"total, stationary", "moving"};
    for (int n = 0; n < 3; n++) {
        auto & name = names[n];

        changing_pm["avg "+name+" organisms size"]     = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n].size,                nullptr};
        changing_pm["avg "+name+" organisms lifetime"] = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n]._organism_lifetime , nullptr};
        changing_pm["avg "+name+" organisms age"]      = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n]._organism_age,       nullptr};
        for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
            changing_pm["avg "+name+" organisms "+ORGANISM_BLOCK_NAMES[i]+" blocks"] = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n].block_avgs[i], nullptr};
        }
        changing_pm["avg "+name+" organisms brain mutation rate"]   = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n].brain_mutation_rate,   nullptr};
        changing_pm["avg "+name+" organisms anatomy mutation rate"] = ChangingReturn{ValueType::DOUBLE, &ic->avgs[n].anatomy_mutation_rate, nullptr};
        changing_pm[name+" organisms amount"] = ChangingReturn{ValueType::INT64, nullptr, &ic->avgs[n].total};
    }
}

const std::vector<std::string> & ParametersList::get_changeable_parameters_list() {
    return changeable_pm_list;
}

const std::vector<std::string> & ParametersList::get_changing_parameters_list() {
    return changing_pm_list;
};

ChangeableReturn ParametersList::get_changeable_value_address_from_name(std::string & name) {
    return changeable_pm[name];
}

ChangingReturn ParametersList::get_changing_value_address_from_name(std::string & name) {
    return changing_pm[name];
}
