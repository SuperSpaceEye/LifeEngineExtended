//
// Created by spaceeye on 27.06.22.
//

#include "MiscFuncs.h"

int display_dialog_message(const std::string &message, bool disable_warnings) {
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