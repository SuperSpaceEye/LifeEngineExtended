//
// Created by spaceeye on 07.05.2022.
//

#include "WindowCore.h"
//
//void WindowCore::closeEvent(QCloseEvent *event) {
//    cp.stop_engine = true;
//    while (!wait_for_engine_to_pause()) {}
//    std::cout << "here\n";
//    QWidget::closeEvent(event);
//}

bool WindowCore::eventFilter(QObject *watched, QEvent *event) {
    //https://doc.qt.io/qt-5/qt.html#MouseButton-enum
    if (event->type() == QEvent::MouseMove) {
        auto mouse_event = dynamic_cast<QMouseEvent *>(event);
        if (right_mouse_button_pressed) {
            int delta_x = mouse_event->x() - last_mouse_x;
            int delta_y = mouse_event->y() - last_mouse_y;

            move_center(delta_x, delta_y);

            last_mouse_x = mouse_event->x();
            last_mouse_y = mouse_event->y();
        }
    } else if (event->type() == QEvent::MouseButtonPress){
        auto mouse_event = dynamic_cast<QMouseEvent*>(event);
        if (mouse_event->button() == Qt::RightButton) {
            auto position = mouse_event->pos();

            if (_ui.simulation_graphicsView->underMouse()) {right_mouse_button_pressed = true;}

            last_mouse_x = position.x();
            last_mouse_y = position.y();
        } else if (mouse_event->button() == Qt::LeftButton) {
            left_mouse_button_pressed = true;

            //if(_ui.simulation_graphicsView->underMouse()) {std::cout << "on widget\n";}
        }
    } else if (event->type() == QEvent::MouseButtonRelease) {
        auto mouse_event = dynamic_cast<QMouseEvent*>(event);
        if (mouse_event->button() == Qt::RightButton) {
            right_mouse_button_pressed = false;
        } else if (mouse_event->button() == Qt::LeftButton) {
            left_mouse_button_pressed = false;
        }
    } else if (event->type() == QEvent::Resize) {
        auto resize_event = dynamic_cast<QResizeEvent*>(event);
        QWidget::resizeEvent(resize_event);
        this->resize(this->parentWidget()->width(), this->parentWidget()->height());
    } else if (event->type() == QEvent::Close) {
        auto close_event = dynamic_cast<QCloseEvent*>(event);
        //closeEvent(close_event);
        cp.stop_engine = true;
        while (!wait_for_engine_to_pause()) {}
        close_event->accept();
    }
    return false;
}

void WindowCore::wheelEvent(QWheelEvent *event) {
    if (_ui.simulation_graphicsView->underMouse()) {
        if (event->delta() > 0) {
            scaling_zoom /= scaling_coefficient;
        } else {
            scaling_zoom *= scaling_coefficient;
        }
    }
}