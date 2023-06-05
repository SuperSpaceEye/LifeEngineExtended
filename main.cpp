#include <QApplication>
#include <QStyleFactory>

#include "MainWindow/MainWindow.h"

int main(int argc, char *argv[]) {
    std::cout << "Loading...\n";
    QApplication app(argc, argv);
    QApplication::setStyle(QStyleFactory::create("WindowsVista"));

    auto window = MainWindow("LifeEngineExtended", {800, 800}, {1000, 200});
    window.show();

    return QApplication::exec();
}