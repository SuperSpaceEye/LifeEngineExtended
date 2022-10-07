#include <emscripten.h>

#include <QApplication>
#include <QStyleFactory>
#include "Source/UIWindows/MainWindow/MainWindow.h"

EMSCRIPTEN_KEEPALIVE
int main() {
    int argc; char *argv[0];
    std::cout << "Loading...\n";
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("WindowsVista"));
    QWidget widget;

    auto window = MainWindow{&widget};
    widget.show();
    return app.exec();
}