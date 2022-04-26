#include "LifeEngineCpp/UIWindow.h"
#include <iostream>
#include "chrono"
#include "boost/unordered_map.hpp"
#include "LifeEngineCpp/Organisms/Organism_parts/OrganismBlock.h"
#include "LifeEngineCpp/BlockTypes.h"
#include "thread"
#include "LifeEngineCpp/Linspace.h"

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 800;
int SIMULATION_WIDTH = 600;
int SIMULATION_HEIGHT = 600;
int MAX_WINDOW_FPS = 60;
int MAX_SIMULATION_FPS = 60;
// should not be num_processes > cpu_cores, cpu_cores-2 works the best (for me)
int START_SIMULATION_THREADS = 14;

//TODO refractor types to be more concrete int->int32
int main() {
    auto window = UIWindow{WINDOW_WIDTH,
                           WINDOW_HEIGHT,
                           SIMULATION_WIDTH,
                           SIMULATION_HEIGHT,
                           MAX_WINDOW_FPS,
                           MAX_SIMULATION_FPS,
                           START_SIMULATION_THREADS};

    window.main_loop();

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
