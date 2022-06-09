//
// Created by spaceeye on 27.03.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONPARAMETERS_H
#define THELIFEENGINECPP_SIMULATIONPARAMETERS_H

struct SimulationParameters {
    //evolution controls
        float food_production_probability = 0.08;
        int   produce_food_every_n_life_ticks = 1;
        int   lifespan_multiplier = 100;
        int   look_range = 50;
        int   auto_produce_food_every_n_ticks = 0;
        int   auto_produce_n_food = 0;
        int   extra_reproduction_cost = 0;
        float global_anatomy_mutation_rate = 0.05;
        float global_brain_mutation_rate = 0.1;
        float killer_damage_amount = 1;

        int   min_reproducing_distance = 1;
        int   max_reproducing_distance = 5;

        float anatomy_mutations_rate_mutation_modifier = 0.01;
        float anatomy_min_possible_mutation_rate = 0.01;
        float anatomy_mutation_rate_delimiter = 0.5;

        float brain_mutation_rate_mutation_modifier = 0.01;
        float brain_min_possible_mutation_rate = 0.1;
        float brain_mutation_rate_delimiter = 0.5;

        int   max_move_range = 5;
        int   min_move_range = 1;
        float move_range_delimiter = 0.5;
        bool  set_fixed_move_range = false;
        int   min_organism_size = 1;

        //Probabilities of creating child with doing:
            int add_cell = 33;
            int change_cell = 33;
            int remove_cell = 33;

        bool reproduction_rotation_enabled = true;
        bool on_touch_kill = false;
        bool use_anatomy_evolved_mutation_rate = true;
        bool use_brain_evolved_mutation_rate = true;
        bool movers_can_produce_food = false;
        bool food_blocks_reproduction = true;
    //world controls
        bool reset_on_total_extinction = true;
        bool pause_on_total_extinction = false;
        bool clear_walls_on_reset = false;
        bool generate_random_walls_on_reset = false;
        bool reproduction_distance_fixed = false;
        bool runtime_rotation_enabled = true;
        bool organism_self_blocks_block_sight = false;
        bool failed_reproduction_eats_food = true;
        bool rotate_every_move_tick = false;
        bool apply_damage_directly = true;
        bool multiply_food_production_prob = false;
        bool simplified_food_production = false;
        bool stop_when_one_food_generated = false;
        bool eat_then_produce = true;
};

#endif //THELIFEENGINECPP_SIMULATIONPARAMETERS_H
