//
// Created by spaceeye on 27.03.2022.
//

#ifndef THELIFEENGINECPP_PARAMETERSSTRUCT_H
#define THELIFEENGINECPP_PARAMETERSSTRUCT_H

struct ParametersStruct {
    //simulation parameters
    //evolution controls
        float food_production_probability = 5;
        int lifespan_multiplier = 100;
        int look_range = 50;
        int auto_food_drop_rate = 0;
        int extra_reproduction_cost = 0;
        float global_mutation_rate = 5;
        //Probabilities
            int add_cell = 33;
            int change_cell = 33;
            int remove_cell = 33;

        bool rotation_enabled = false;
        bool one_touch_kill = true;
        bool use_evolved_mutation_rate = true;
        bool movers_can_produce_food = false;
        bool food_blocks_reproduction = true;
    //world controls
        bool reset_on_total_extinction = true;
        bool pause_on_total_extinction = false;
        bool clear_walls_on_reset = false;
};

#endif //THELIFEENGINECPP_PARAMETERSSTRUCT_H
