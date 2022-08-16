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