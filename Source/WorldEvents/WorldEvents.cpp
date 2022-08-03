//
// Created by spaceeye on 01.08.22.
//

#include "WorldEvents.h"

WorldEvents::WorldEvents(Ui::MainWindow *parent_ui): parent_ui(parent_ui) {
    ui.setupUi(this);
}

void WorldEvents::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_info_window->setChecked(false);
    QWidget::closeEvent(event);
}