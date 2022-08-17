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