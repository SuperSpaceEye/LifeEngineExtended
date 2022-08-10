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
        "==========",
        "Mouth Block food cost modifier",
        "Mouth Block life point amount",
        "Mouth Block lifetime weight",
        "Mouth Block chance weight",
        "==========",
        "Producer Block food cost modifier",
        "Producer Block life point amount",
        "Producer Block lifetime weight",
        "Producer Block chance weight",
        "==========",
        "Mover Block food cost modifier",
        "Mover Block life point amount",
        "Mover Block lifetime weight",
        "Mover Block chance weight",
        "==========",
        "Killer Block food cost modifier",
        "Killer Block life point amount",
        "Killer Block lifetime weight",
        "Killer Block chance weight",
        "==========",
        "Armor Block food cost modifier",
        "Armor Block life point amount",
        "Armor Block lifetime weight",
        "Armor Block chance weight",
        "==========",
        "Eye Block food cost modifier",
        "Eye Block life point amount",
        "Eye Block lifetime weight",
        "Eye Block chance weight"
    };


    changeable_pm["None"] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};
    changeable_pm["=========="] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};

    changeable_pm["food production probability"]   = ChangeableReturn{ValueType::FLOAT, &sp->food_production_probability, nullptr};
    changeable_pm["lifespan multiplier"]           = ChangeableReturn{ValueType::FLOAT, &sp->lifespan_multiplier, nullptr};
    changeable_pm["extra reproduction cost"]       = ChangeableReturn{ValueType::FLOAT, &sp->extra_reproduction_cost, nullptr};
    changeable_pm["extra mover reproductive cost"] = ChangeableReturn{ValueType::FLOAT, &sp->extra_mover_reproductive_cost, nullptr};
    changeable_pm["global anatomy mutation rate"]  = ChangeableReturn{ValueType::FLOAT, &sp->global_anatomy_mutation_rate, nullptr};
    changeable_pm["global brain mutation rate"]    = ChangeableReturn{ValueType::FLOAT, &sp->global_brain_mutation_rate, nullptr};
    changeable_pm["killer damage amount"]          = ChangeableReturn{ValueType::FLOAT, &sp->killer_damage_amount, nullptr};

    changeable_pm["anatomy mutation rate delimiter"] = ChangeableReturn{ValueType::FLOAT, &sp->anatomy_mutation_rate_delimiter, nullptr};
    changeable_pm["brain mutation rate delimiter"]   = ChangeableReturn{ValueType::FLOAT, &sp->brain_mutation_rate_delimiter, nullptr};

    changeable_pm["auto produce food every n ticks"] = ChangeableReturn{ValueType::INT, nullptr, &sp->auto_produce_food_every_n_ticks};
    changeable_pm["auto produce n food"]             = ChangeableReturn{ValueType::INT, nullptr, &sp->auto_produce_n_food};
    changeable_pm["produce food every n life ticks"] = ChangeableReturn{ValueType::INT, nullptr, &sp->produce_food_every_n_life_ticks};
    changeable_pm["add cell"]                        = ChangeableReturn{ValueType::INT, nullptr, &sp->add_cell};
    changeable_pm["change cell"]                     = ChangeableReturn{ValueType::INT, nullptr, &sp->change_cell};
    changeable_pm["remove cell"]                     = ChangeableReturn{ValueType::INT, nullptr, &sp->remove_cell};

    changeable_pm["Mouth Block food cost modifier"]= ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Mouth Block life point amount"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Mouth Block lifetime weight"]   = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Mouth Block chance weight"]     = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Producer Block food cost modifier"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Producer Block life point amount"]  = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Producer Block lifetime weight"]    = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Producer Block chance weight"]      = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Mover Block food cost modifier"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Mover Block life point amount"]  = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Mover Block lifetime weight"]    = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Mover Block chance weight"]      = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Killer Block food cost modifier"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Killer Block life point amount"]  = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Killer Block lifetime weight"]    = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Killer Block chance weight"]      = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Armor Block food cost modifier"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Armor Block life point amount"]  = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Armor Block lifetime weight"]    = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Armor Block chance weight"]      = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changeable_pm["Eye Block food cost modifier"] = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.food_cost_modifier, nullptr};
    changeable_pm["Eye Block life point amount"]  = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.life_point_amount, nullptr};
    changeable_pm["Eye Block lifetime weight"]    = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.lifetime_weight, nullptr};
    changeable_pm["Eye Block chance weight"]      = ChangeableReturn{ValueType::FLOAT, &bp->MouthBlock.chance_weight, nullptr};

    changing_pm_list = std::vector<std::string> {
            "None",
            "==========",
            "avg total organisms size",
            "avg total organisms lifetime",
            "avg total organisms age",
            "avg total organisms mouth cells",
            "avg total organisms producer cells",
            "avg total organisms mover cells",
            "avg total organisms killer cells",
            "avg total organisms armor cells",
            "avg total organisms eye cells",
            "avg total organisms brain mutation rate",
            "avg total organisms anatomy mutation rate",
            "total organisms amount",
            "==========",
            "avg stationary organisms size",
            "avg stationary organisms lifetime",
            "avg stationary organisms age",
            "avg stationary organisms mouth cells",
            "avg stationary organisms producer cells",
            "avg stationary organisms mover cells",
            "avg stationary organisms killer cells",
            "avg stationary organisms armor cells",
            "avg stationary organisms eye cells",
            "avg stationary organisms brain mutation rate",
            "avg stationary organisms anatomy mutation rate",
            "stationary organisms amount",
            "==========",
            "moving organisms with eyes",
            "avg moving organisms size",
            "avg moving organisms lifetime",
            "avg moving organisms age",
            "avg moving organisms mouth cells",
            "avg moving organisms producer cells",
            "avg moving organisms mover cells",
            "avg moving organisms killer cells",
            "avg moving organisms armor cells",
            "avg moving organisms eye cells",
            "avg moving organisms brain mutation rate",
            "avg moving organisms anatomy mutation rate",
            "total organisms amount",
    };

    changing_pm["None"] = ChangingReturn{ValueType::NONE, nullptr, nullptr};
    changeable_pm["=========="] = ChangeableReturn{ValueType::NONE, nullptr, nullptr};

    changing_pm["moving organisms with eyes"] = ChangingReturn{ValueType::INT64, nullptr, &ic->organisms_with_eyes};

    changing_pm["avg total organisms size"]                  = ChangingReturn{ValueType::DOUBLE, &ic->total_avg.size,                  nullptr};
    changing_pm["avg total organisms lifetime"]              = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._organism_lifetime,    nullptr};
    changing_pm["avg total organisms age"]                   = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._organism_age,         nullptr};
    changing_pm["avg total organisms mouth cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._mouth_blocks,         nullptr};
    changing_pm["avg total organisms producer cells"]        = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._producer_blocks,      nullptr};
    changing_pm["avg total organisms mover cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._mover_blocks,         nullptr};
    changing_pm["avg total organisms killer cells"]          = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._killer_blocks,        nullptr};
    changing_pm["avg total organisms armor cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._armor_blocks,         nullptr};
    changing_pm["avg total organisms eye cells"]             = ChangingReturn{ValueType::DOUBLE, &ic->total_avg._eye_blocks,           nullptr};
    changing_pm["avg total organisms brain mutation rate"]   = ChangingReturn{ValueType::DOUBLE, &ic->total_avg.brain_mutation_rate,   nullptr};
    changing_pm["avg total organisms anatomy mutation rate"] = ChangingReturn{ValueType::DOUBLE, &ic->total_avg.anatomy_mutation_rate, nullptr};
    changing_pm["total organisms amount"]                    = ChangingReturn{ValueType::INT64, nullptr, &ic->total_avg.total};

    changing_pm["avg stationary organisms size"]                  = ChangingReturn{ValueType::DOUBLE, &ic->station_avg.size,                  nullptr};
    changing_pm["avg stationary organisms lifetime"]              = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._organism_lifetime,    nullptr};
    changing_pm["avg stationary organisms age"]                   = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._organism_age,         nullptr};
    changing_pm["avg stationary organisms mouth cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._mouth_blocks,         nullptr};
    changing_pm["avg stationary organisms producer cells"]        = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._producer_blocks,      nullptr};
    changing_pm["avg stationary organisms mover cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._mover_blocks,         nullptr};
    changing_pm["avg stationary organisms killer cells"]          = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._killer_blocks,        nullptr};
    changing_pm["avg stationary organisms armor cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._armor_blocks,         nullptr};
    changing_pm["avg stationary organisms eye cells"]             = ChangingReturn{ValueType::DOUBLE, &ic->station_avg._eye_blocks,           nullptr};
    changing_pm["avg stationary organisms brain mutation rate"]   = ChangingReturn{ValueType::DOUBLE, &ic->station_avg.brain_mutation_rate,   nullptr};
    changing_pm["avg stationary organisms anatomy mutation rate"] = ChangingReturn{ValueType::DOUBLE, &ic->station_avg.anatomy_mutation_rate, nullptr};
    changing_pm["stationary organisms amount"]                    = ChangingReturn{ValueType::INT64, nullptr, &ic->station_avg.total};

    changing_pm["avg moving organisms size"]                  = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg.size,                  nullptr};
    changing_pm["avg moving organisms lifetime"]              = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._organism_lifetime,    nullptr};
    changing_pm["avg moving organisms age"]                   = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._organism_age,         nullptr};
    changing_pm["avg moving organisms mouth cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._mouth_blocks,         nullptr};
    changing_pm["avg moving organisms producer cells"]        = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._producer_blocks,      nullptr};
    changing_pm["avg moving organisms mover cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._mover_blocks,         nullptr};
    changing_pm["avg moving organisms killer cells"]          = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._killer_blocks,        nullptr};
    changing_pm["avg moving organisms armor cells"]           = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._armor_blocks,         nullptr};
    changing_pm["avg moving organisms eye cells"]             = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg._eye_blocks,           nullptr};
    changing_pm["avg moving organisms brain mutation rate"]   = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg.brain_mutation_rate,   nullptr};
    changing_pm["avg moving organisms anatomy mutation rate"] = ChangingReturn{ValueType::DOUBLE, &ic->moving_avg.anatomy_mutation_rate, nullptr};
    changing_pm["moving organisms amount"]                    = ChangingReturn{ValueType::INT64, nullptr, &ic->moving_avg.total};
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
