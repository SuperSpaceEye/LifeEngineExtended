#include "LifeEngineCpp/WindowCore.h"
#include <QApplication>

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 800;
int SIMULATION_WIDTH = 600;
int SIMULATION_HEIGHT = 600;
int MAX_WINDOW_FPS = 60;
int MAX_SIMULATION_FPS = 0;
// should not be num_processes > cpu_cores, cpu_cores-3 works the best (for me)
int START_SIMULATION_THREADS = 13;

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QWidget widget;

    auto window = WindowCore{WINDOW_WIDTH,
                             WINDOW_HEIGHT,
                             SIMULATION_WIDTH,
                             SIMULATION_HEIGHT,
                             MAX_WINDOW_FPS,
                             MAX_SIMULATION_FPS,
                             START_SIMULATION_THREADS,
                             &widget};
    widget.show();
    return app.exec();

//    std::string my_num;
//
//    std::cin >> my_num;
//
//    double num = 0;
//    try {
//        num = boost::lexical_cast<double>(my_num);
//    } catch(boost::bad_lexical_cast) {
//        std::cout << "bad_cast\n";
//    }
//
//    std::cout << num;

//    boost::unordered_map<int, boost::unordered_map<int, int>> test_map;
//    boost::unordered_map<int, boost::unordered_map<int, int>> second_map;
//
//    int width = 6;
//    int height = 6;
//
//    auto point = std::chrono::high_resolution_clock::now();
//    for (int x = 0; x < width; x++) {
//        for (int y = 0; y < height; y++) {
//            test_map[x][y] = x * y;
//        }
//    }
//
//    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - point).count() << std::endl;
//
//    point = std::chrono::high_resolution_clock::now();
//    for (auto const &xmap: test_map) {
//        for (auto const &yxmap: xmap.second) {
//            second_map[xmap.first].count(yxmap.first);
//        }
//    }
//    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - point).count() << std::endl;
//
//    for (int x = 0; x < width; x++) {
//        for (int y = 0; y < height; y++) {
//            second_map[x][y] = x * y;
//        }
//    }
//
//    point = std::chrono::high_resolution_clock::now();
//    for (auto const &xmap: test_map) {
//        for (auto const &yxmap: xmap.second) {
//            second_map[xmap.first].count(yxmap.first);
//        }
//    }
//    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - point).count() << std::endl;
//    auto test_vector = std::vector<int>{};
//    test_vector.resize(width*height);
//
//    for (int x = 0; x < 5; x++) {
//        for (int y = 0; y < 5; y++) {
//            for (int i = 0; i < width*height; i++) {
//                //if (test_vector[width * relative_x + relative_y] > 0) { continue; }
//                volatile auto val = test_vector[width*x+y];
//            }
//        }
//    }
//
//    point = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < width*height; i++) {
//        volatile auto val = test_vector[i];
//    }
//    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - point).count() << std::endl;
}
