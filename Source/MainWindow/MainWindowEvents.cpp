// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 07.05.2022.
//

#include "MainWindow.h"

bool MainWindow::eventFilter(QObject *watched, QEvent *event) {
    //https://doc.qt.io/qt-5/qt.html#MouseButton-enum
    switch (event->type()) {
        case QEvent::MouseMove: {
            auto mouse_event = dynamic_cast<QMouseEvent *>(event);
            if (wheel_mouse_button_pressed) {
                int delta_x = mouse_event->x() - last_mouse_x_pos;
                int delta_y = mouse_event->y() - last_mouse_y_pos;

                move_center(delta_x, delta_y);
            }
            last_mouse_x_pos = mouse_event->x();
            last_mouse_y_pos = mouse_event->y();
        } break;
        case QEvent::MouseButtonPress: {
            auto mouse_event = dynamic_cast<QMouseEvent*>(event);
            auto position = mouse_event->pos();
            if (mouse_event->button() == Qt::MiddleButton) {
                wheel_mouse_button_pressed = true;
            } else if (mouse_event->button() == Qt::RightButton) {
                right_mouse_button_pressed = true;
            } else if (mouse_event->button() == Qt::LeftButton) {
                left_mouse_button_pressed = true;
            }
            last_mouse_x_pos = position.x();
            last_mouse_y_pos = position.y();

            change_main_simulation_grid = _ui.simulation_graphicsView->underMouse();
            change_editing_grid = ee._ui.editor_graphicsView->underMouse();

            //Removes focus from line edits, buttons, etc. so that user can use keyboard buttons without problems.
            if (_ui.simulation_graphicsView->underMouse()) {setFocus();}
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
            if (watched->objectName().toStdString() == "QWidgetClassWindow") {
                ecp.stop_engine = true;
                while (!wait_for_engine_to_pause_force()) {}
                exit(0);
            }
            close_event->accept();
        } break;
        default: break;
    }

    process_keyboard_events();
    last_event_execution = clock_now();

    return false;
}

void MainWindow::process_keyboard_events() {
    float mult;
    if (SHIFT_pressed) { mult = SHIFT_keyboard_movement_multiplier;} else { mult = 1;}
    mult *= std::chrono::duration_cast<std::chrono::microseconds>(clock_now() - last_event_execution).count()/1000.;
    if (W_pressed) { center_y -= keyboard_movement_amount * scaling_zoom * mult;}
    if (S_pressed) { center_y += keyboard_movement_amount * scaling_zoom * mult;}
    if (D_pressed) { center_x += keyboard_movement_amount * scaling_zoom * mult;}
    if (A_pressed) { center_x -= keyboard_movement_amount * scaling_zoom * mult;}
}

void MainWindow::wheelEvent(QWheelEvent *event) {
    if (_ui.simulation_graphicsView->underMouse()) {
        if (event->delta() > 0) {
            scaling_zoom /= scaling_coefficient;
        } else {
            scaling_zoom *= scaling_coefficient;
        }
    }
}

void MainWindow::keyPressEvent(QKeyEvent * event) {
    switch (event->key()) {
        case Qt::Key_M: {
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

        //TODO refactor?
        case Qt::Key_Up:
        case Qt::Key_W: W_pressed = true;break;
        case Qt::Key_Down:
        case Qt::Key_S: S_pressed = true;break;
        case Qt::Key_Left:
        case Qt::Key_A: A_pressed = true;break;
        case Qt::Key_Right:
        case Qt::Key_D: D_pressed = true;break;

        case Qt::Key_Shift: SHIFT_pressed = true;break;

        case Qt::Key_R: reset_scale_view();break;
        case Qt::Key_Space: _ui.tb_pause->setChecked(!ecp.tb_paused);break;
        case Qt::Key_Period: b_pass_one_tick_slot();

        case Qt::Key_Plus: scaling_zoom /= scaling_coefficient;break;
        case Qt::Key_Minus: scaling_zoom *= scaling_coefficient;break;

        case Qt::Key_1: rb_food_slot();_ui.rb_food->setChecked(true);break;
        case Qt::Key_2: rb_kill_slot();_ui.rb_kill->setChecked(true);break;
        case Qt::Key_3: rb_wall_slot();_ui.rb_wall->setChecked(true);break;
        case Qt::Key_4: ee.rb_place_organism_slot();ee._ui.rb_place_organism->setChecked(true);break;
        case Qt::Key_5: ee.rb_choose_organism_slot();ee._ui.rb_chose_organism->setChecked(true);break;

    }
}

void MainWindow::keyReleaseEvent(QKeyEvent * event) {
    switch (event->key()) {
        case Qt::Key_M: allow_menu_hidden_change = true;break;

        case Qt::Key_Up:
        case Qt::Key_W: W_pressed = false;break;
        case Qt::Key_Down:
        case Qt::Key_S: S_pressed = false;break;
        case Qt::Key_Left:
        case Qt::Key_A: A_pressed = false;break;
        case Qt::Key_Right:
        case Qt::Key_D: D_pressed = false;break;

        case Qt::Key_Shift: SHIFT_pressed = false;break;
    }
}