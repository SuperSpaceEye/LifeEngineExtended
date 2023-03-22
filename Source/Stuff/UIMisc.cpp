// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 27.06.22.
//

#include "UIMisc.h"

bool display_dialog_message(const std::string &message, bool disable_warnings) {
    if (disable_warnings) {return true;}
    DescisionMessageBox msg{"Warning", QString::fromStdString(message), "OK", "Cancel"};
    return msg.exec();
}

bool display_save_type_dialog_message(int &result, bool use_lfew) {
    QDialog main_dialog;

    result = 2;
    std::string str = use_lfew ? "lfew" : "lfeo";

    auto lfew = new QPushButton(QString::fromStdString("Save as " + str));
    QObject::connect(lfew, &QPushButton::clicked, [&result, &main_dialog]{
        result = 0;
        main_dialog.accept();
    });

    auto json = new QPushButton("Save as json");
    QObject::connect(json, &QPushButton::clicked, [&result, &main_dialog]{
        result = 1;
        main_dialog.accept();
    });

    auto cancel = new QPushButton("Cancel");
    QObject::connect(cancel, &QPushButton::clicked, [&result, &main_dialog]{
        result = 2;
        main_dialog.accept();
    });

    QDialogButtonBox dialog(&main_dialog);
    dialog.addButton(lfew,   QDialogButtonBox::AcceptRole);
    dialog.addButton(json,   QDialogButtonBox::AcceptRole);
    dialog.addButton(cancel, QDialogButtonBox::AcceptRole);

    dialog.setCenterButtons(true);

    dialog.setMinimumSize(500, 100);

    main_dialog.setFixedSize(500, 100);

    auto * screen = QGuiApplication::primaryScreen();
    auto screen_geometry = screen->geometry();
    int width = screen_geometry.width();
    int height = screen_geometry.height();

    main_dialog.move(width/2-250, height/2-50-20);

    return main_dialog.exec();
}

void display_message(const std::string &message) {
    QMessageBox msg;
    msg.setText(QString::fromStdString(message));
    msg.setWindowTitle("Warning");
    msg.exec();
}

std::string convert_num_bytes(uint64_t num_bytes) {
    if (!(num_bytes/1024)) {return std::to_string(num_bytes) + " B";}
    double d_num_bytes = num_bytes/1024;
    if(!uint64_t(d_num_bytes/1024)) {return to_str(d_num_bytes, 1) + "KiB";}
    d_num_bytes /= 1024;
    if(!uint64_t(d_num_bytes/1024)) {return to_str(d_num_bytes, 1) + "MiB";}
    d_num_bytes /= 1024;
    if(!uint64_t(d_num_bytes/1024)) {return to_str(d_num_bytes, 1) + "GiB";}
    d_num_bytes /= 1024;
    return to_str(d_num_bytes, 1) + "TiB";
}

void clear_console() {
#if defined _WIN32
    system("cls");
    //clrscr(); // including header file : conio.h
#elif defined (__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
    system("clear");
#endif
}

std::string convert_seconds(uint64_t num_seconds) {
    std::string return_str;

    std::string seconds;
    std::string minutes;
    std::string hours;

    if (num_seconds/60/60 > 0) {
        auto result = num_seconds/60/60;
        hours += std::to_string(result) + "h ";
        num_seconds -= result * 60 * 60;
    }

    if (num_seconds/60 > 0) {
        auto result = num_seconds/60;
        minutes += std::to_string(result) + "m ";
        num_seconds -= result * 60;
    }

    seconds += std::to_string(num_seconds) + "s ";

    return_str += hours + minutes + seconds;

    return return_str;
}

bool choose_node_window(NodeType &new_node_type) {
    QDialog main_dialog;

    new_node_type = NodeType::Conditional;
    auto change_value_button = new QPushButton("Change Value Node");
    QObject::connect(change_value_button, &QPushButton::clicked, [&new_node_type, &main_dialog](){
        new_node_type = NodeType::ChangeValue;
        main_dialog.accept();
    });

    auto conditional_button = new QPushButton("Conditional Node");
    QObject::connect(conditional_button, &QPushButton::clicked, [&new_node_type, &main_dialog](){
        new_node_type = NodeType::Conditional;
        main_dialog.accept();
    });

    QDialogButtonBox dialog(&main_dialog);
    dialog.addButton(change_value_button, QDialogButtonBox::AcceptRole);
    dialog.addButton(conditional_button, QDialogButtonBox::AcceptRole);

    dialog.setCenterButtons(true);

    dialog.setMinimumSize(400, 100);

    main_dialog.setFixedSize(400, 100);

    auto * screen = QGuiApplication::primaryScreen();
    auto screen_geometry = screen->geometry();
    int width = screen_geometry.width();
    int height = screen_geometry.height();

    main_dialog.move(width/2-200, height/2-50-20);

    return main_dialog.exec();
}

#ifdef __CUDA_USED__
#include "get_device_count.cuh"
#endif

bool cuda_is_available() {
#ifdef __CUDA_USED__
    return get_device_count();
#else
    return false;
#endif
}

jmp_buf env;

void on_sigabrt(int signum) {
    signal (signum, SIG_DFL);
    longjmp (env, 1);
}