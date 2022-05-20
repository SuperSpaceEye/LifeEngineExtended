#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}
