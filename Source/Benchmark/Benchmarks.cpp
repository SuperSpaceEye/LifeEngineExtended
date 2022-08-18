//
// Created by spaceeye on 16.08.22.
//

#include "Benchmarks.h"

Benchmarks::Benchmarks(Ui::MainWindow &parent_window): parent_window(parent_window) {
    ui.setupUi(this);
}

void Benchmarks::closeEvent(QCloseEvent *event) {
    parent_window.tb_open_benchmarks->setChecked(false);
    QWidget::closeEvent(event);
}

void Benchmarks::benchmark_buttons_enabled(bool state) {
    ui.b_benchmark_produce_food     ->setEnabled(state);
    ui.b_benchmark_eat_food         ->setEnabled(state);
    ui.b_benchmark_apply_damage     ->setEnabled(state);
    ui.b_benchmark_tick_lifetime    ->setEnabled(state);
    ui.b_benchmark_erase_organisms  ->setEnabled(state);
    ui.b_benchmark_reserve_organisms->setEnabled(state);
    ui.b_benchmark_get_observations ->setEnabled(state);
    ui.b_benchmark_think_decision   ->setEnabled(state);
    ui.b_benchmark_rotate_organism  ->setEnabled(state);
    ui.b_benchmark_move_organism    ->setEnabled(state);
    ui.b_benchmark_try_make_child   ->setEnabled(state);
    ui.b_run_all_benchmarks         ->setEnabled(state);
}

void Benchmarks::update_result_info() {
    std::string result_str;
    auto & results = benchmark.get_results();

    if (results.empty()) {
        result_str.append("No benchmark data.");
        ui.benchmarks_output_text_edit->setText(QString::fromStdString(result_str));
        return;
    }

    for (auto & result: results) {
        std::string benchmark_type;
        switch (result.benchmark_type) {
            case BenchmarkTypes::ProduceFood:      benchmark_type = "produce food"; break;
            case BenchmarkTypes::EatFood:          benchmark_type = " "; break;
            case BenchmarkTypes::ApplyDamage:      benchmark_type = " "; break;
            case BenchmarkTypes::TickLifetime:     benchmark_type = " "; break;
            case BenchmarkTypes::EraseOrganisms:   benchmark_type = " "; break;
            case BenchmarkTypes::ReserveOrganisms: benchmark_type = " "; break;
            case BenchmarkTypes::GetObservations:  benchmark_type = " "; break;
            case BenchmarkTypes::ThinkDecision:    benchmark_type = " "; break;
            case BenchmarkTypes::RotateOrganism:   benchmark_type = " "; break;
            case BenchmarkTypes::MoveOrganism:     benchmark_type = " "; break;
            case BenchmarkTypes::TryMakeChild:     benchmark_type = " "; break;
        }
        std::string avg_time;
        if (result.num_tried > 0) {
            avg_time = "Avg nanoseconds per operation: " + std::to_string(result.total_time_measured / result.num_tried);
        }

        result_str.append("Benchmark type: ").append(benchmark_type).append(" ||| Num organism in benchmark: ")
        .append(std::to_string(result.num_organisms)).append(" ||| Num iterations: ").append(std::to_string(result.num_iterations))
        .append(" ||| Operations measured: ").append(std::to_string(result.num_tried)).append(" ||| Additional data: ")
        .append(result.additional_data).append(" ||| ").append(avg_time).append("\n\n\n");
    }
    ui.benchmarks_output_text_edit->setText(QString::fromStdString(result_str));
}