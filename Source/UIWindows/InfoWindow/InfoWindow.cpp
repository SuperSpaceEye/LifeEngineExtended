//
// Created by spaceeye on 25.07.22.
//

#include "InfoWindow.h"

InfoWindow::InfoWindow(Ui::MainWindow * parent_ui) {
    _ui.setupUi(this);
    _parent_ui = parent_ui;

    auto button = new QPushButton("Return", this);
    QPushButton::connect(button, &QPushButton::clicked, [&](){ this->close();});
    this->layout()->addWidget(button);
}

void InfoWindow::closeEvent(QCloseEvent *event) {
    _parent_ui->tb_open_info_window->setChecked(false);
    QWidget::closeEvent(event);
}