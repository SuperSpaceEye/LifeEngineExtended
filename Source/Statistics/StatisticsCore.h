// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 24.06.22.
//

#ifndef THELIFEENGINECPP_STATISTICSCORE_H
#define THELIFEENGINECPP_STATISTICSCORE_H
#include "StatisticsUI.h"
#include "../MainWindow/WindowUI.h"

class StatisticsCore: public QWidget {
    Q_OBJECT
public:
    Ui::Statistics _ui;
    Ui::MainWindow * _parent_ui = nullptr;
    StatisticsCore()=default;
    void init(Ui::MainWindow * parent_ui) {
        _ui.setupUi(this);
        _parent_ui = parent_ui;
    }

    void closeEvent(QCloseEvent * event) override {
        _parent_ui->tb_open_statisctics->setChecked(false);
        QWidget::closeEvent(event);
    }
};

#endif //THELIFEENGINECPP_STATISTICSCORE_H
