#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>
#include <QStyleFactory>

#ifndef __CUDA_USED__
#define __CUDA_USED__=0
#endif

#if defined(__WIN32)
#include <windows.h>
#endif

//#if __CUDA_USED__
//#include "LifeEngineCpp/SimulationEngineModes/SimulationEngineCuda.cuh"
//#endif

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
