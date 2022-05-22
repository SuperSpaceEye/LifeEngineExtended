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
    switch (event->type()) {
        case QEvent::MouseMove: {
            auto mouse_event = dynamic_cast<QMouseEvent *>(event);
            if (wheel_mouse_button_pressed) {
                int delta_x = mouse_event->x() - last_mouse_x;
                int delta_y = mouse_event->y() - last_mouse_y;

                move_center(delta_x, delta_y);
            }
            last_mouse_x = mouse_event->x();
            last_mouse_y = mouse_event->y();
        } break;
        case QEvent::MouseButtonPress: {
            auto mouse_event = dynamic_cast<QMouseEvent*>(event);
            auto position = mouse_event->pos();
            if (mouse_event->button() == Qt::MiddleButton) {
                wheel_mouse_button_pressed = true;
                change_main_simulation_grid = true;
                last_mouse_x = position.x();
                last_mouse_y = position.y();
            } else if (mouse_event->button() == Qt::RightButton) {
                right_mouse_button_pressed = true;
            } else if (mouse_event->button() == Qt::LeftButton) {
                left_mouse_button_pressed = true;
            }
            last_mouse_x = position.x();
            last_mouse_y = position.y();

            change_main_simulation_grid = _ui.simulation_graphicsView->underMouse();
            change_editing_grid = _ui.simulation_graphicsView->underMouse();
        } break;
        case QEvent::MouseButtonRelease: {
            auto mouse_event = dynamic_cast<QMouseEvent*>(event);
            if (mouse_event->button() == Qt::MiddleButton) {
                wheel_mouse_button_pressed = false;
            } else if (mouse_event->button() == Qt::RightButton) {
                right_mouse_button_pressed = false;
            } else if (mouse_event->button() == Qt::LeftButton) {
                left_mouse_button_pressed = false;
            }

            change_editing_grid = false;
            change_main_simulation_grid = false;
        } break;
        case QEvent::Resize: {
            auto resize_event = dynamic_cast<QResizeEvent*>(event);
            QWidget::resizeEvent(resize_event);
            this->resize(this->parentWidget()->width(), this->parentWidget()->height());
        } break;
        case QEvent::Close: {
            auto close_event = dynamic_cast<QCloseEvent*>(event);
            cp.stop_engine = true;
            while (!wait_for_engine_to_pause()) {}
            close_event->accept();
        } break;
        default: break;
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

void WindowCore::keyPressEvent(QKeyEvent * event) {
    if (event->key() == Qt::Key_M) {
        if (_ui.simulation_graphicsView->underMouse()) {
            if (allow_menu_hidden_change) {
                if (!menu_hidden) {
                    _ui.menu_frame->hide();
                    menu_hidden = true;
                    allow_menu_hidden_change = false;
                } else {
                    _ui.menu_frame->show();
                    menu_hidden = false;
                    allow_menu_hidden_change = false;
                }
            }
        }
    }
}

void WindowCore::keyReleaseEvent(QKeyEvent * event) {
    if (event->key() == Qt::Key_M) {
        allow_menu_hidden_change = true;
    }
}