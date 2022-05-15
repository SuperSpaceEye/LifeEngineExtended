//
// Created by spaceeye on 27.03.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONPARAMETERS_H
#define THELIFEENGINECPP_SIMULATIONPARAMETERS_H

struct SimulationParameters {
    //simulation sp
    //evolution controls
        float food_production_probability = 0.05;
        int   lifespan_multiplier = 100;
        int   look_range = 50;
        int   auto_food_drop_rate = 0;
        int   extra_reproduction_cost = 0;
        float global_mutation_rate = 0.05;
        float killer_damage_amount = 1;

        int min_reproducing_distance = 0;
        int max_reproducing_distance = 3;
        //Probabilities of creating child with doing:
            int add_cell = 25;
            int change_cell = 25;
            int remove_cell = 25;

        bool reproduction_rotation_enabled = true;
        bool one_touch_kill = false;
        bool use_evolved_mutation_rate = true;
        bool movers_can_produce_food = false;
        bool food_blocks_reproduction = true;
    //world controls
        bool reset_on_total_extinction = false;
        bool pause_on_total_extinction = false;
        bool clear_walls_on_reset = false;
        bool generate_random_walls_on_reset = false;
        bool reproduction_distance_fixed = false;
        bool runtime_rotation_enabled = false;
};

#endif //THELIFEENGINECPP_SIMULATIONPARAMETERS_H
