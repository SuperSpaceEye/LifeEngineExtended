//
// Created by spaceeye on 27.03.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONPARAMETERS_H
#define THELIFEENGINECPP_SIMULATIONPARAMETERS_H

struct SimulationParameters {
    //simulation sim_parameters
    //evolution controls
        float food_production_probability = 0.5;
        int lifespan_multiplier = 100;
        int look_range = 50;
        int auto_food_drop_rate = 0;
        int extra_reproduction_cost = 0;
        float global_mutation_rate = 5;
        //Probabilities
            int add_cell = 25;
            int change_cell = 25;
            int remove_cell = 25;
            int do_nothing = 25;

        bool rotation_enabled = false;
        bool one_touch_kill = true;
        bool use_evolved_mutation_rate = true;
        bool movers_can_produce_food = false;
        bool food_blocks_reproduction = true;
    //world controls
        bool reset_on_total_extinction = true;
        bool pause_on_total_extinction = false;
        bool clear_walls_on_reset = false;
        bool generate_random_walls_on_reset = false;
};

#endif //THELIFEENGINECPP_SIMULATIONPARAMETERS_H
