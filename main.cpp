#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>
#include <QStyleFactory>

#ifndef __CUDA_USED__
#define __CUDA_USED__=0
#endif

//#if __CUDA_USED__
//#include "LifeEngineCpp/SimulationEngineModes/SimulationEngineCuda.cuh"
//#endif

int main(int argc, char *argv[]) {
//#if __CUDA_USED__
//    auto engine = SimulationEngineCuda();
//    engine.cuda_tick();
//    std::cout << "here\n";
//#endif

    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("WindowsVista"));
    QWidget widget;

    auto window = WindowCore{&widget};
    widget.show();
    return app.exec();
}
