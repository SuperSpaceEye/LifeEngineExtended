#include "LifeEngineCpp/UIWindow.h"
//#include <iostream>
//#include "chrono"
//#include "boost/unordered_map.hpp"
//#include "LifeEngineCpp/BlockTypes.h"
//#include "thread"
//#include "engine_mutex"

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 800;
int SIMULATION_WIDTH = 600;
int SIMULATION_HEIGHT = 600;
int MAX_WINDOW_FPS = 60;
int MAX_SIMULATION_FPS = 0;

int main() {
    auto window = UIWindow{WINDOW_WIDTH,
                           WINDOW_HEIGHT,
                           SIMULATION_WIDTH,
                           SIMULATION_HEIGHT,
                           MAX_WINDOW_FPS,
                           MAX_SIMULATION_FPS};

    window.main_loop();
}
