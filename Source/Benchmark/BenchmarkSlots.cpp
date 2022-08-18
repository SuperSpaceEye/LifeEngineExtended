//
// Created by spaceeye on 17.08.22.
//

#include "Benchmarks.h"

void Benchmarks::b_benchmark_produce_food_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ProduceFood});
}

void Benchmarks::b_benchmark_apply_damage_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ApplyDamage});
}

void Benchmarks::b_benchmark_eat_food_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::EatFood});
}

void Benchmarks::b_benchmark_erase_organisms_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::EraseOrganisms});
}

void Benchmarks::b_benchmark_get_observations_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::GetObservations});
}

void Benchmarks::b_benchmark_move_organism_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::MoveOrganism});
}

void Benchmarks::b_benchmark_reserve_organism_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ReserveOrganisms});
}

void Benchmarks::b_benchmark_rotate_organism_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::RotateOrganism});
}

void Benchmarks::b_benchmark_think_decision_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ThinkDecision});
}

void Benchmarks::b_benchmark_tick_lifetime_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::TickLifetime});
}

void Benchmarks::b_benchmark_try_make_child_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::TryMakeChild});
}

void Benchmarks::b_run_all_benchmarks_slot() {
    benchmark_buttons_enabled(false);
    benchmark.init_benchmark();
    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ProduceFood,
                                                             BenchmarkTypes::ApplyDamage,
                                                             BenchmarkTypes::EatFood,
                                                             BenchmarkTypes::EraseOrganisms,
                                                             BenchmarkTypes::GetObservations,
                                                             BenchmarkTypes::MoveOrganism,
                                                             BenchmarkTypes::RotateOrganism,
                                                             BenchmarkTypes::ThinkDecision,
                                                             BenchmarkTypes::TickLifetime,
//                                                             BenchmarkTypes::TryMakeChild
    });
}

void Benchmarks::b_stop_benchmarks_slot() {
    benchmark_buttons_enabled(true);
    benchmark.finish_benchmarking();
    update_result_info();
}
