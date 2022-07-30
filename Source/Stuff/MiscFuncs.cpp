// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 27.06.22.
//

#include "MiscFuncs.h"

bool display_dialog_message(const std::string &message, bool disable_warnings) {
    if (disable_warnings) {return true;}
    DescisionMessageBox msg{"Warning", QString::fromStdString(message), "OK", "Cancel"};
    return msg.exec();
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