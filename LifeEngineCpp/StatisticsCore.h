//
// Created by spaceeye on 24.06.22.
//

#ifndef THELIFEENGINECPP_STATISTICSCORE_H
#define THELIFEENGINECPP_STATISTICSCORE_H

#include "StatisticsUI.h"
#include "WindowUI.h"

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

    void closeEvent(QCloseEvent * event) {
        _parent_ui->tb_open_statisctics->setChecked(false);
        QWidget::closeEvent(event);
    }
};

#endif //THELIFEENGINECPP_STATISTICSCORE_H
