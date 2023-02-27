//
// Created by spaceeye on 17.08.22.
//

#include "Benchmarks.h"

void Benchmarks::b_benchmark_produce_food_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ProduceFood});
}

void Benchmarks::b_benchmark_apply_damage_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ApplyDamage});
}

void Benchmarks::b_benchmark_eat_food_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::EatFood});
}

void Benchmarks::b_benchmark_get_observations_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::GetObservations});
}

void Benchmarks::b_benchmark_move_organism_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::MoveOrganism});
}

void Benchmarks::b_benchmark_reserve_organism_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ReserveOrganisms});
}

void Benchmarks::b_benchmark_rotate_organism_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::RotateOrganism});
}

void Benchmarks::b_benchmark_think_decision_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ThinkDecision});
}

void Benchmarks::b_benchmark_tick_lifetime_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::TickLifetime});
}

void Benchmarks::b_benchmark_try_make_child_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::TryMakeChild});
}

void Benchmarks::b_run_all_benchmarks_slot() {
//    benchmark_buttons_enabled(false);
//    benchmark.init_benchmark();
//    benchmark.start_benchmarking(std::vector<BenchmarkTypes>{BenchmarkTypes::ProduceFood,
//                                                             BenchmarkTypes::ApplyDamage,
//                                                             BenchmarkTypes::EatFood,
//                                                             BenchmarkTypes::GetObservations,
//                                                             BenchmarkTypes::MoveOrganism,
//                                                             BenchmarkTypes::RotateOrganism,
//                                                             BenchmarkTypes::ThinkDecision,
//                                                             BenchmarkTypes::TickLifetime,
//                                                             BenchmarkTypes::TryMakeChild
//    });
}

void Benchmarks::b_stop_benchmarks_slot() {
//    benchmark_buttons_enabled(true);
//    benchmark.finish_benchmarking();
//    update_result_info();
}

void Benchmarks::b_apply_grid_size_slot() {
//    benchmark.resize_benchmark_grid(new_width, new_height);
}


void Benchmarks::le_grid_width_slot() {
    le_slot_lower_bound<int>(new_width, new_width, "int", ui.le_grid_width, 100, "100");
}

void Benchmarks::le_grid_height_slot() {
    le_slot_lower_bound<int>(new_height, new_height, "int", ui.le_grid_width, 100, "100");
}

void Benchmarks::le_num_benchmark_organisms_slot() {
//    auto temp = benchmark.get_num_organisms();
//    le_slot_lower_bound<int>(temp, temp, "int", ui.le_num_benchmark_organisms, 1, "1");
//    benchmark.set_num_organisms(temp);
}

void Benchmarks::le_num_iterations_slot() {
//    auto temp = benchmark.get_total_num_iterations();
//    le_slot_lower_bound<int>(temp, temp, "int", ui.le_num_iterations, 10, "10");
//    benchmark.set_num_tries(temp);
}

void Benchmarks::le_num_organisms_diameter_slot() {
//    auto temp = benchmark.get_organisms_diameter();
//    le_slot_lower_bound<int>(temp, temp, "int", ui.le_organisms_diameter, 1, "1");
//    benchmark.set_organisms_diameter(temp);
}