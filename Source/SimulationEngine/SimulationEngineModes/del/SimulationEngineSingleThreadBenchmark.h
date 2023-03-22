//
// Created by spaceeye on 16.08.22.
//

#ifndef LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
#define LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H

#include <chrono>
#include <thread>
#include <random>

#include "../SimulationEngineSingleThread.h"
#include "../../OrganismsController.h"
#include "../../../Containers/OrganismBlockParameters.h"

enum class BenchmarkTypes {
    ProduceFood,
    EatFood,
    ApplyDamage,
    TickLifetime,
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

    int num_iterations = 200;
    int num_organisms = 10'000;
    int organisms_diameter = 1;

    boost::unordered_map<BenchmarkTypes, std::vector<OrganismContainer>> benchmark_organisms;

    bool initialized = false;
    volatile bool benchmark_running = false;
    volatile bool stop_benchmark = false;

    std::vector<BenchmarkResult> benchmark_results;

    void create_benchmark_organisms();
    void remove_benchmark_organisms();

    void prepare_produce_food_benchmark();
    void benchmark_produce_food(BenchmarkResult &res, bool flag);

    void prepare_eat_food_benchmark();
    void benchmark_eat_food(BenchmarkResult &res, bool flag);

    void benchmark_apply_damage(BenchmarkResult &res, bool flag);

    void prepare_tick_lifetime_benchmark();
    void benchmark_tick_lifetime(BenchmarkResult &res, bool flag);

    void benchmark_get_observations(BenchmarkResult &res, bool flag);

    void benchmark_think_decision(BenchmarkResult &res, bool flag);

    void benchmark_rotate_organism(BenchmarkResult &res, bool flag);

    void benchmark_move_organism(BenchmarkResult &res, bool flag);

    void prepare_try_make_child_benchmark(int num_organisms);
    void benchmark_try_make_child(BenchmarkResult &res, bool flag);

    void place_organisms_of_type(Organism *organism, int num_organisms,
                                 BenchmarkResult &result, int additional_distance);
    void reset_state();
public:
    SimulationEngineSingleThreadBenchmark();

    void init_benchmark();

    bool benchmark_is_running() const {return benchmark_running;}

    bool set_seed(uint64_t seed) {if (!benchmark_running) {gen.set_seed(seed); return true;} else {return false;}}
    bool resize_benchmark_grid(int width, int height);
    bool set_num_organisms(int num) {if (!benchmark_running) {num_organisms = num; return true;} else {return false;}}
    bool set_num_tries(int num) {if (!benchmark_running) {num_iterations = num; return true;} else {return false;}}

    void start_benchmarking(const std::vector<BenchmarkTypes>& benchmarks_to_do);

    void finish_benchmarking();

    void stop_benchmark_thread();

    const std::vector<BenchmarkResult> & get_results();
    int get_total_num_iterations() const {return num_iterations;}
    int get_num_organisms() const {return num_organisms;}
    void set_organisms_diameter(int diameter) {organisms_diameter = diameter;}
    int get_organisms_diameter() const {return organisms_diameter;}
};


#endif //LIFEENGINEEXTENDED_SIMULATIONENGINESINGLETHREADBENCHMARK_H
