#include <QApplication>
#include <QStyleFactory>
#include "LifeEngineCpp/WindowCore.h"

#if defined(__WIN32)
#include <windows.h>
#endif

int main(int argc, char *argv[]) {
#if defined(__WIN32)
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif

    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("WindowsVista"));
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}
