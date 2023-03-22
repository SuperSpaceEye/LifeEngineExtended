// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 27.06.22.
//

#ifndef THELIFEENGINECPP_MISCFUNCS_H
#define THELIFEENGINECPP_MISCFUNCS_H

#include <string>
#include <sstream>
#include <iomanip>
#include <csetjmp>
#include <csignal>

#include <QDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>
#include <QLineEdit>
#include <QObject>
#include <QDialogButtonBox>
#include <QGuiApplication>
#include <QScreen>

#include "UIWindows/WorldEvents/EventNodes.h"
#include "UIWindows/WorldEvents/Misc/WorldEventsEnums.h"

template<typename T>
struct result_struct {
    bool is_valid;
    T result;
};

template <typename T> std::string to_str(const T& t, int float_precision = 2) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(float_precision) << t;
    return stream.str();
}

class DescisionMessageBox : public QDialog {
Q_OBJECT

private:
    QVBoxLayout *vertical_layout;
    QHBoxLayout *horizontal_layout;
    QPushButton *accept_button;
    QPushButton *decline_button;
    QLabel *content_label;
public:
    DescisionMessageBox(const QString& title, const QString& content,
                        const QString& accept_text, const QString& decline_text, QWidget* parent=0)
            : QDialog(parent) {
        vertical_layout = new QVBoxLayout();
        horizontal_layout = new QHBoxLayout();
        accept_button = new QPushButton(accept_text, this);
        decline_button = new QPushButton(decline_text, this);
        content_label = new QLabel(content, this);

        setLayout(vertical_layout);
        vertical_layout->addWidget(content_label, 2);
        vertical_layout->addLayout(horizontal_layout, 1);
        horizontal_layout->addWidget(accept_button);
        horizontal_layout->addWidget(decline_button);

        connect(accept_button, &QPushButton::pressed, this, &QDialog::accept);
        connect(decline_button, &QPushButton::pressed, this, &QDialog::reject);

        this->setWindowTitle(title);
    }
};

bool display_dialog_message(const std::string &message, bool disable_warnings);

bool display_save_type_dialog_message(int &result, bool use_lfew = true);

void display_message(const std::string &message);

template<typename T>
result_struct<T> try_convert_string(const std::string & str) {
    if (str.empty()) {
        return result_struct<T>{false, static_cast<T>(0)};
    }

    bool is_valid = true;

    for (auto chr: str) {
        std::string s;
        s += chr;
        if (s!="0"&&s!="1"&&s!="2"&&s!="3"&&s!="3"&&s!="4"&&s!="5"&&s!="6"&&s!="7"&&s!="8"&&s!="9"&&s!="."&&s!="-") {
            is_valid = false;
            break;
        }
    }
    if (!is_valid) {
        return result_struct<T>{false, static_cast<T>(0)};
    }

    return result_struct<T>{true, static_cast<T>(std::stod(str))};
}

template<typename T>
result_struct<T> try_convert_message_box_template(const std::string& message, QLineEdit *line_edit, T &fallback_value) {
    result_struct<T> result = try_convert_string<T>(line_edit->text().toStdString());

    if (result.is_valid) {
        return result;
    }

    display_message(message);
    line_edit->setText(QString::fromStdString(to_str(fallback_value, 5)));
    return result;
}

template<typename T>
void le_slot_no_bound(T & _fallback,
                      T & to_change,
                      const std::string & type,
                      QLineEdit * lineEdit) {
    T fallback = _fallback;
    auto result = try_convert_message_box_template<T>("Input is not a type " + type + ".", lineEdit, fallback);
    if (!result.is_valid) {return;}
    to_change = result.result;
}

template<typename T>
void le_slot_lower_bound(T & _fallback,
                         T & to_change,
                         const std::string & type,
                         QLineEdit * lineEdit,
                         T lower_bound,
                         const std::string & lower_bound_string) {
    T fallback = _fallback;
    auto result = try_convert_message_box_template<T>("Input is not a type " + type + ".", lineEdit, fallback);
    if (!result.is_valid) {return;}
    if (result.result < lower_bound) {display_message("Input cannot be less than " + lower_bound_string + "."); return;}
    to_change = result.result;
}

template<typename T>
void le_slot_lower_upper_bound(T & _fallback,
                               T & to_change,
                               const std::string & type,
                               QLineEdit * lineEdit,
                               T lower_bound,
                               const std::string & lower_bound_string,
                               T upper_bound,
                               const std::string & upper_bound_string) {
    T fallback = _fallback;
    auto result = try_convert_message_box_template<T>("Input is not a type " + type + ".", lineEdit, fallback);
    if (!result.is_valid) {return;}
    if (result.result < lower_bound) {display_message("Input cannot be less than " + lower_bound_string + "."); return;}
    if (result.result > upper_bound) {display_message("Input cannot be more than " + upper_bound_string + "."); return;}
    to_change = result.result;
}

template<typename T>
void le_slot_lower_lower_bound(T & _fallback,
                               T & to_change,
                               const std::string & type,
                               QLineEdit * lineEdit,
                               T lower_bound,
                               const std::string & lower_bound_string,
                               T lower2_bound,
                               const std::string & lower2_bound_string) {
    T fallback = _fallback;
    auto result = try_convert_message_box_template<T>("Input is not a type " + type + ".", lineEdit, fallback);
    if (!result.is_valid) {return;}
    if (result.result <  lower_bound) {display_message("Input cannot be less than " + lower_bound_string  + "."); return;}
    if (result.result < lower2_bound) {display_message("Input cannot be less than " + lower2_bound_string + "."); return;}
    to_change = result.result;
}

std::string convert_num_bytes(uint64_t num_bytes);

void clear_console();

std::string convert_seconds(uint64_t num_seconds);

bool choose_node_window(NodeType &new_node_type);

bool cuda_is_available();

void on_sigabrt(int signum);

template <typename ...A>
bool try_and_catch_abort(std::function<void(A...)> f, A... args)
{
    extern jmp_buf env;
    if (setjmp (env) == 0) {
        signal(SIGABRT, &on_sigabrt);
        f(args...);
        signal (SIGABRT, SIG_DFL);
        return false;
    }
    else {
        return true;
    }
}

#endif //THELIFEENGINECPP_MISCFUNCS_H
