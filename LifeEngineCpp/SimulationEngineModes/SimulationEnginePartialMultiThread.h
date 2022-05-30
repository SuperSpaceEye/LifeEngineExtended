//
// Created by spaceeye on 16.05.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONENGINEPARTIALMULTITHREAD_H
#define THELIFEENGINECPP_SIMULATIONENGINEPARTIALMULTITHREAD_H

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
//#include "EagerWorkerPartial.h"
#include "../GridBlocks/BaseGridBlock.h"
#include "../Organism/CPU/Organism.h"
#include "../BlockTypes.hpp"
#include "../Containers/CPU/EngineControlContainer.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Containers/CPU/OrganismBlockParameters.h"
#include "../Linspace.h"
#include "SimulationEngineSingleThread.h"

struct EngineDataContainer;
struct eager_worker_partial;

enum class PartialSimulationStage {
    PlaceOrganisms,
    ProduceFood,
    EatFood,
    ApplyDamage,
    TickLifetime,
    //EraseOrganisms,
    //ReserveObservations,
    GetObservations,
    ThinkDecision,
    //DoDecision,
    //TryMakeChild
};

class SimulationEnginePartialMultiThread {
private:
    static void place_organism (EngineDataContainer * dc, Organism * organism);

    static void eat_food        (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void tick_lifetime(EngineDataContainer *dc, std::vector<std::vector<int>> &to_erase,
                              Organism *organism, int thread_num, int organism_pos);

    static void erase_organisms (EngineDataContainer * dc, std::vector<std::vector<int>>& to_erase);

    static void get_observations(EngineDataContainer *dc, Organism *&organism,
                                 std::vector<std::vector<Observation>> &organism_observations,
                                 SimulationParameters *sp, int organism_num);
    static void calculate_threads_points(int num_points, int num_threads, std::vector<std::vector<int>> & thread_points);

    static void start_stage(EngineDataContainer *dc, PartialSimulationStage stage,
                            std::vector<std::vector<int>> &thread_points);
public:
    static void
    partial_multi_thread_tick(EngineDataContainer *dc, EngineControlParameters *cp,
                              OrganismBlockParameters *bp, SimulationParameters *sp,
                              boost::mt19937 *mt);

    static void build_threads(EngineDataContainer &dc, EngineControlParameters &cp,
                              SimulationParameters &sp);

    static void kill_threads(EngineDataContainer &dc);

    static void thread_tick(PartialSimulationStage stage, EngineDataContainer *dc,
                            SimulationParameters *sp, boost::mt19937 *mt, int start_pos,
                            int end_pos, int thread_num);

};

struct eager_worker_partial {
    EngineDataContainer * dc = nullptr;
    SimulationParameters * sp = nullptr;
    int start_pos = 0;
    int end_pos = 0;
    int thread_num;

    std::random_device r;
    boost::mt19937 mt;

    eager_worker_partial() = default;
    eager_worker_partial(EngineDataContainer * dc, SimulationParameters * sp, int thread_num):
            dc(dc), sp(sp), thread_num(thread_num){
        auto seed = std::seed_seq{r(),r(),r(),r(),r(),r(),r(),r()};
        mt = boost::mt19937(seed);
        thread = std::thread{&eager_worker_partial::main_loop, this};
    }
    eager_worker_partial(const eager_worker_partial & worker):
            dc(worker.dc), sp(worker.sp), thread_num(worker.thread_num){
        auto seed = std::seed_seq{r(),r(),r(),r(),r(),r(),r(),r()};
        mt = boost::mt19937(seed);
        thread = std::thread{&eager_worker_partial::main_loop, this};
    }

    inline void work(PartialSimulationStage stage, int start, int stop) {
        start_pos = start;
        end_pos = stop;
        thread_stage = stage;
        has_work = true;
    }

    inline void finish() {
        while (has_work) {}
    }

    inline void stop_work() {
        exiting = true;
    }

    inline ~eager_worker_partial() {stop_thread();}
    inline void stop_thread() {
        exiting = true;
        has_work = true;
        if (thread.joinable()) {
            thread.join();
        }
    }
private:
    std::atomic<bool> has_work{false};

    std::atomic<bool> exiting{false};

    PartialSimulationStage thread_stage;

//    __attribute__((optimize("O0")))
//    bool wait_for_work() {
//        while (!has_work) {
//            if (exiting) {
//                return true;
//            }
//        }
//        return false;
//    }

//    __attribute__((optimize("O0")))
    void main_loop() {
        while (true) {
//            if (wait_for_work()) { return;}
            while (!has_work.load()) {
                if (exiting.load()) {
                    return;
                }
            }
            SimulationEnginePartialMultiThread::thread_tick(thread_stage, dc, sp, &mt, start_pos, end_pos, thread_num);
            has_work = false;
        }
    }

    std::thread thread;
};


#endif //THELIFEENGINECPP_SIMULATIONENGINEPARTIALMULTITHREAD_H
