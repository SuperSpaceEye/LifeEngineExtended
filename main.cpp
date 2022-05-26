#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("WindowsVista"));
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}
