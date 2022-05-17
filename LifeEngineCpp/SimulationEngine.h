//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_LIFEENGINE_H
#define LANGUAGES_LIFEENGINE_H

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#include "GridBlocks/BaseGridBlock.h"
#include "Organism/Organism.h"
#include "BlockTypes.hpp"
#include "EngineControlContainer.h"
#include "EngineDataContainer.h"
#include "OrganismBlockParameters.h"
#include "Linspace.h"
#include "SimulationEngineModes/SimulationEnginePartialMultiThread.h"


//TODO move simulation grid translation to here
class SimulationEngine {
    EngineControlParameters& cp;
    EngineDataContainer& dc;
    OrganismBlockParameters& op;
    SimulationParameters& sp;

    std::mutex& mutex;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;

    void process_user_action_pool(){};

    void simulation_tick();
    void partial_multi_threaded_tick();
    void multi_threaded_tick();
    void cuda_tick();

    void change_mode();


    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist;

public:
    SimulationEngine(EngineDataContainer& engine_data_container, EngineControlParameters& engine_control_parameters,
                     OrganismBlockParameters& organism_block_parameters, SimulationParameters& simulation_parameters,
                     std::mutex& mutex);
    void threaded_mainloop();

    static void tick_of_single_thread();

};

//struct eager_worker_partial {
//    EngineDataContainer * dc = nullptr;
//    int start_relative_x = 0;
//    int start_relative_y = 0;
//    int end_relative_x = 0;
//    int end_relative_y = 0;
//
//    std::random_device rd;
//    std::mt19937 mt;
//
//    eager_worker_partial() = default;
//    eager_worker_partial(EngineDataContainer * dc, int start_relative_x, int start_relative_y, int end_relative_x, int end_relative_y):
//            dc(dc), start_relative_x(start_relative_x), start_relative_y(start_relative_y), end_relative_x(end_relative_x), end_relative_y(end_relative_y){
//        mt = std::mt19937(rd());
//    }
//    eager_worker_partial(const eager_worker_partial & worker):
//            dc(worker.dc), start_relative_x(worker.start_relative_x), start_relative_y(worker.start_relative_y),
//            end_relative_x(worker.end_relative_x), end_relative_y(worker.end_relative_y){
//        mt = std::mt19937(rd());
//    }
//
//    inline void work() {
//        has_work = true;
//    }
//
//    inline void finish() {
//        while (has_work) {}
//    }
//
//    inline void stop_work() {
//        has_work = true;
//        exiting = true;
//    }
//
//    inline eager_worker_partial_partial() { stop_thread(); }
//    inline void stop_thread() {
//        exiting = true;
//        has_work = true;
//        if (thread.joinable()) {
//            thread.join();
//        }
//    }
//private:
//    bool has_work = false;
//
//    bool exiting = false;
//    bool thread_started = false;
//
//    std::thread thread = std::thread([this] {
//        thread_started = true;
//        while (true) {
//            while (!has_work) {
//                if (exiting) {
//                    return;
//                }
//            }
//            SimulationEngine::single_threaded_tick(dc, &mt, start_relative_x, start_relative_y, end_relative_x, end_relative_y);
//            has_work = false;
//        }
//    });
//};

#endif //LANGUAGES_LIFEENGINE_H
