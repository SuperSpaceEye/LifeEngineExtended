//
// Created by spaceeye on 16.08.22.
//

#ifndef LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
#define LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H

#include <chrono>
#include <thread>
#include <random>

#include "SimulationEngineSingleThread.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"

enum class BenchmarkTypes {
    ProduceFood,
    EatFood,
    ApplyDamage,
    TickLifetime,
    EraseOrganisms,
    ReserveOrganisms,
    GetObservations,
    ThinkDecision,
    RotateOrganism,
    MoveOrganism,
    TryMakeChild
};

struct BenchmarkResult {
    int num_organisms = 0;
    int num_iterations = 0;
    uint64_t num_tried = 0;
    uint64_t total_time_measured = 0;
    BenchmarkTypes benchmark_type;
    std::string additional_data;
};

struct OrganismContainer {
    std::string additional_data;
    Organism * organism;
    int additional_distance = 0;
};

class SimulationEngineSingleThreadBenchmark {
    EngineDataContainer dc;
    SimulationParameters sp;
    OrganismBlockParameters bp;
    std::thread benchmark_thread;

    uint64_t seed = 0;
    lehmer64 gen{0};

    int num_iterations = 500;
    int num_organisms = 10'000;

    boost::unordered_map<BenchmarkTypes, std::vector<OrganismContainer>> benchmark_organisms;

    bool initialized = false;
    volatile bool benchmark_running = false;
    volatile bool stop_benchmark = false;

    std::vector<BenchmarkResult> benchmark_results;

    void create_benchmark_organisms();
    void remove_benchmark_organisms();

    void prepare_produce_food_benchmark();
    void benchmark_produce_food(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_eat_food_benchmark();
    void benchmark_eat_food(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_apply_damage_benchmark();
    void benchmark_apply_damage(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_tick_lifetime_benchmark();
    void benchmark_tick_lifetime(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_erase_organisms_benchmark(Organism *organism);
    void benchmark_erase_organisms(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_reserve_organisms_benchmark();
    void benchmark_reserve_organisms();

    void prepare_get_observations_benchmark();
    void benchmark_get_observations(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_think_decision_benchmark();
    void benchmark_think_decision(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_rotate_organism_benchmark();
    void benchmark_rotate_organism(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_move_organism_benchmark();
    void benchmark_move_organism(bool randomized_organism_access, BenchmarkResult &res);

    void prepare_try_make_child_benchmark();
    void benchmark_try_make_child(bool randomized_organism_access, BenchmarkResult &res);
    void finish_try_make_child_benchmark();

    void place_organisms_of_type(Organism *organism, int num_organisms,
                                 BenchmarkResult &result, int additional_distance);
    void reset_state();
public:
    SimulationEngineSingleThreadBenchmark();

    void init_benchmark();

    bool benchmark_is_running() const {return benchmark_running;}

    bool set_seed(uint64_t seed);
    bool resize_benchmark_grid(int width, int height);
    bool set_num_organisms(int num);
    bool set_num_tries(int num);

    void start_benchmarking(const std::vector<BenchmarkTypes>& benchmarks_to_do);

    void finish_benchmarking();

    void stop_benchmark_thread();

    const std::vector<BenchmarkResult> & get_results();
    int get_total_num_iterations() const {return num_iterations;}
};


#endif //LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H