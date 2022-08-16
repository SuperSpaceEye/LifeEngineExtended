//
// Created by spaceeye on 16.08.22.
//

#ifndef LIFEENGINEEXTENDED_BENCHMARKS_H
#define LIFEENGINEEXTENDED_BENCHMARKS_H

#include "BenchmarkUI.h"
#include "../MainWindow/WindowUI.h"

class Benchmarks: public QWidget {
    Q_OBJECT
private:
    Ui::Benchmark ui{};
    Ui::MainWindow & parent_window;

    void closeEvent(QCloseEvent * event) override;
public:
    Benchmarks(Ui::MainWindow & parent_window);

};


#endif //LIFEENGINEEXTENDED_BENCHMARKS_H
