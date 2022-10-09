//
// Created by spaceeye on 25.07.22.
//

#ifndef LIFEENGINEEXTENDED_INFOWINDOW_H
#define LIFEENGINEEXTENDED_INFOWINDOW_H

#include "InfoWindowUI.h"
#include <QKeyEvent>
#include "../MainWindow/WindowUI.h"

class InfoWindow: public QWidget {
Q_OBJECT
private:
    Ui::Info _ui{};
    Ui::MainWindow * _parent_ui = nullptr;

    void closeEvent(QCloseEvent * event) override;
    void keyPressEvent(QKeyEvent * event) override {
        if (event->key() == Qt::Key_Escape) {
            close();
        }
    }
public:
    InfoWindow(Ui::MainWindow * parent_ui);
};


#endif //LIFEENGINEEXTENDED_INFOWINDOW_H
