// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 24.06.22.
//

#ifndef THELIFEENGINECPP_STATISTICSCORE_H
#define THELIFEENGINECPP_STATISTICSCORE_H

#include <QKeyEvent>

#include "StatisticsUI.h"
#include "../MainWindow/WindowUI.h"

class StatisticsCore: public QWidget {
    Q_OBJECT
public:
    Ui::Statistics ui{};
    Ui::MainWindow * parent_ui = nullptr;
    StatisticsCore()=default;
    StatisticsCore(Ui::MainWindow * parent_ui): parent_ui(parent_ui) {
        ui.setupUi(this);
    };

    void closeEvent(QCloseEvent * event) override {
        parent_ui->tb_open_statisctics->setChecked(false);
        QWidget::closeEvent(event);
    }

    void keyPressEvent(QKeyEvent * event) override {
        if (event->key() == Qt::Key_Escape) {
            close();
        }
    }
};

#endif //THELIFEENGINECPP_STATISTICSCORE_H
