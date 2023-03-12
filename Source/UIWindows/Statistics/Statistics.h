// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 24.06.22.
//

#ifndef THELIFEENGINECPP_STATISTICSCORE_H
#define THELIFEENGINECPP_STATISTICSCORE_H

#include "UIWindows/MainWindow/WindowUI.h"

#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/UIMisc.h"
#include "Containers/OrganismInfoContainer.h"

#include "StatisticsUI.h"

class Statistics: public QWidget {
    Q_OBJECT
private:
    std::array<std::array<QLabel*, NUM_ORGANISM_BLOCKS>, 3> labels;

    void make_organism_blocks_labels();
public:
    Ui::Statistics ui;
    Ui::MainWindow * parent_ui = nullptr;
    Statistics()=default;
    Statistics(Ui::MainWindow * parent_ui): parent_ui(parent_ui) {
        ui.setupUi(this);
        make_organism_blocks_labels();
    };

    void closeEvent(QCloseEvent * event) override {
        parent_ui->tb_open_statisctics->setChecked(false);
        QWidget::closeEvent(event);
    }

    void update_statistics(const OrganismInfoContainer &info, EngineDataContainer & edc, int float_precision, float scaling_zoom, float center_x, float center_y);
};

#endif //THELIFEENGINECPP_STATISTICSCORE_H
