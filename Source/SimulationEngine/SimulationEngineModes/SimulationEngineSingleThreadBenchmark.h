//
// Created by spaceeye on 16.08.22.
//

#ifndef LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
#define LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H

#include <chrono>
#include <thread>

#include "SimulationEngineSingleThread.h"

class SimulationEngineSingleThreadBenchmark {
    EngineDataContainer dc;
    SimulationParameters sp;

    std::vector<uint64_t> seeds;
    lehmer64 gen{0};

    int num_tries = 10'000;
    int num_organisms = 10'000;

    boost::unordered_map<std::string, std::vector<Organism *>> benchmark_organisms;

    bool initialized = false;

    void create_benchmark_organisms();
    void remove_benchmark_organisms();

    void prepare_produce_food();
    void benchmark_produce_food();

    void prepare_eat_food();
    void benchmark_eat_food();

    void prepare_apply_damage();
    void benchmark_apply_damage();

    void prepare_tick_lifetime();
    void benchmark_tick_lifetime();

    void prepare_erase_organisms();
    void benchmark_erase_organisms();

    void prepare_reserve_organisms();
    void benchmark_reserve_organisms();

    void prepare_get_observations();
    void benchmark_get_observations();

    void prepare_think_decision();
    void benchmark_think_decision();

    void prepare_rotate_organism();
    void benchmark_rotate_organism();

    void prepare_move_organism();
    void benchmark_move_organism();

    void prepare_try_make_child();
    void benchmark_try_make_child();
public:
    SimulationEngineSingleThreadBenchmark();

    void init_benchmark();

    void set_seeds(std::vector<uint64_t> & seeds);
    void resize_benchmark_grid();
    void set_num_organisms();
    void set_num_tries();

    void finish_benchmarking();
};


#endif //LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
