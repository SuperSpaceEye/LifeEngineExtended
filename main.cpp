#include <QApplication>
#include <QStyleFactory>
#include "LifeEngineCpp/WindowCore.h"

//TODO add more comments
//TODO add benchmarks

int main(int argc, char *argv[]) {
    std::cout << "Loading...\n";
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("WindowsVista"));
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}