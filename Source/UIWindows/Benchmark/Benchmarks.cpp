//
// Created by spaceeye on 16.08.22.
//

#include "Benchmarks.h"

Benchmarks::Benchmarks(Ui::MainWindow &parent_window): parent_window(parent_window) {
    ui.setupUi(this);
    update_result_info();
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
    ui.b_benchmark_get_observations ->setEnabled(state);
    ui.b_benchmark_think_decision   ->setEnabled(state);
    ui.b_benchmark_rotate_organism  ->setEnabled(state);
    ui.b_benchmark_move_organism    ->setEnabled(state);
    ui.b_benchmark_try_make_child   ->setEnabled(state);
    ui.b_run_all_benchmarks         ->setEnabled(state);
    ui.b_apply_grid_size            ->setEnabled(state);
    ui.le_num_benchmark_organisms   ->setEnabled(state);
    ui.le_num_iterations            ->setEnabled(state);
    ui.le_organisms_diameter        ->setEnabled(state);
}

void Benchmarks::update_() {
//    if (!benchmark.benchmark_is_running()) { benchmark_buttons_enabled(true);}
//    if (!benchmark.benchmark_is_running() && !updated_after_end) {
//        updated_after_end = true;
//        update_result_info();
//    }
//    if (benchmark.benchmark_is_running()) {
//        update_result_info();
//        updated_after_end = false;
//    }
}

void Benchmarks::update_result_info() {
//    std::string result_str;
//    auto & results = benchmark.get_results();

//    if (results.empty()) {
//        result_str.append("No benchmark data.");
//        ui.benchmarks_output_text_edit->setText(QString::fromStdString(result_str));
//        return;
//    }
//
//    BenchmarkTypes last_type = results[0].benchmark_type;
//
//    for (auto & result: results) {
//        std::string benchmark_type;
//        switch (result.benchmark_type) {
//            case BenchmarkTypes::ProduceFood:      benchmark_type = "produce food";      break;
//            case BenchmarkTypes::EatFood:          benchmark_type = "eat food";          break;
//            case BenchmarkTypes::ApplyDamage:      benchmark_type = "apply damage";      break;
//            case BenchmarkTypes::TickLifetime:     benchmark_type = "tick lifetime";     break;
//            case BenchmarkTypes::ReserveOrganisms: benchmark_type = "reserve organisms"; break;
//            case BenchmarkTypes::GetObservations:  benchmark_type = "get observations";  break;
//            case BenchmarkTypes::ThinkDecision:    benchmark_type = "think decision";    break;
//            case BenchmarkTypes::RotateOrganism:   benchmark_type = "rotate organism";   break;
//            case BenchmarkTypes::MoveOrganism:     benchmark_type = "move organism";     break;
//            case BenchmarkTypes::TryMakeChild:     benchmark_type = "try make child";    break;
//        }
//        std::string avg_time;
//        if (result.num_tried > 0) {
//            avg_time = "Avg nanoseconds per operation: " + std::to_string(result.total_time_measured / result.num_tried);
//        }
//
//        if (result.benchmark_type != last_type) {
//            last_type = result.benchmark_type;
//            result_str.append("==============================\n\n\n");
//        }
//
//        result_str.append("Benchmark type: ").append(benchmark_type).append(" ||| Num organism in benchmark: ")
//        .append(std::to_string(result.num_organisms)).append(" ||| Num iterations: ").append(std::to_string(result.num_iterations))
//        .append(" ||| Operations measured: ").append(std::to_string(result.num_tried)).append(" ||| Additional data: ")
//        .append(result.additional_data).append(" ||| ").append(avg_time).append("\n\n\n");
//    }
//    ui.benchmarks_output_text_edit->setText(QString::fromStdString(result_str));
//
//    ui.benchmarks_output_text_edit->verticalScrollBar()->setValue(ui.benchmarks_output_text_edit->verticalScrollBar()->maximum());
}