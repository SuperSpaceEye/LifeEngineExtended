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