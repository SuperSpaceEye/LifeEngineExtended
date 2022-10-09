#include <QApplication>
#include <QStyleFactory>
#include "Source/UIWindows/MainWindow/MainWindow.h"

int main(int argc, char *argv[]) {
    std::cout << "Loading...\n";
    std::cout << "main 0\n";
    QApplication app(argc, argv);
    std::cout << "main 1\n";
    app.setStyle(QStyleFactory::create("WindowsVista"));
    std::cout << "main 2\n";
    QWidget widget;
    std::cout << "main 3\n";

    auto window = MainWindow{&widget};
    std::cout << "main 4\n";
    widget.show();
    std::cout << "main 5\n";
    return app.exec();
}