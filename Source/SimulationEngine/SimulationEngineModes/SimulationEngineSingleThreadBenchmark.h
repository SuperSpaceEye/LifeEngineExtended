//
// Created by spaceeye on 16.08.22.
//

#ifndef LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
#define LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H

#include <chrono>
#include <thread>

#include "SimulationEngineSingleThread.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"

struct BenchmarkResult {
    int total_num_tries = 0;
    int num_organisms = 0;
    uint64_t num_tried = 0;
    uint64_t total_time_measured = 0;
    std::string benchmark_type;
};

struct BenchmarkThreadControls {
    bool stop = false;
    bool pause = true;
};

class SimulationEngineSingleThreadBenchmark {
    EngineDataContainer dc;
    SimulationParameters sp;
    OrganismBlockParameters bp;
    std::thread benchmark_thread;

    std::vector<uint64_t> seeds;
    lehmer64 gen{0};

    int num_tries = 10'000;
    int num_organisms = 10'000;

    boost::unordered_map<std::string, std::vector<Organism *>> benchmark_organisms;

    bool initialized = false;
    bool benchmark_running = false;

    std::vector<BenchmarkResult> benchmark_results;

    void create_benchmark_organisms();
    void remove_benchmark_organisms();

    void prepare_produce_food_benchmark();
    void benchmark_produce_food();
    void finish_produce_food_benchmark();

    void prepare_eat_food_benchmark();
    void benchmark_eat_food();
    void finish_eat_food_benchmark();

    void prepare_apply_damage_benchmark();
    void benchmark_apply_damage();
    void finish_apply_damage_benchmark();

    void prepare_tick_lifetime_benchmark();
    void benchmark_tick_lifetime();
    void finish_tick_lifetime_benchmark();

    void prepare_erase_organisms_benchmark();
    void benchmark_erase_organisms();
    void finish_erase_organisms_benchmark();

    void prepare_reserve_organisms_benchmark();
    void benchmark_reserve_organisms();
    void finish_reserve_organisms_benchmark();

    void prepare_get_observations_benchmark();
    void benchmark_get_observations();
    void finish_get_observations_benchmark();

    void prepare_think_decision_benchmark();
    void benchmark_think_decision();
    void finish_think_decision_benchmark();

    void prepare_rotate_organism_benchmark();
    void benchmark_rotate_organism();
    void finish_rotate_organism_benchmark();

    void prepare_move_organism_benchmark();
    void benchmark_move_organism();
    void finish_move_organism_benchmark();

    void prepare_try_make_child_benchmark();
    void benchmark_try_make_child();
    void finish_try_make_child_benchmark();
public:
    SimulationEngineSingleThreadBenchmark();

    void init_benchmark();

    bool set_seeds(std::vector<uint64_t> & seeds);
    bool resize_benchmark_grid(int width, int height);
    bool set_num_organisms(int num);
    bool set_num_tries(int num);

    void finish_benchmarking();

    const std::vector<BenchmarkResult> & get_results();
};


#endif //LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
