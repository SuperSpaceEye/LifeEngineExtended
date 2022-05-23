#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("windowsvista"));
//    for (auto & key :QStyleFactory::keys().toVector()) {
//        std::cout << key.toStdString() << "\n";
//    }
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}
