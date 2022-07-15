// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

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

#include "../../GridBlocks/BaseGridBlock.h"
#include "../../Organism/CPU/Organism.h"
#include "../../Stuff/BlockTypes.hpp"
#include "../../Containers/CPU/EngineControlContainer.h"
#include "../../Containers/CPU/EngineDataContainer.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "../../Stuff/Linspace.h"
#include "SimulationEngineSingleThread.h"
#include "../../PRNGS/lehmer64.h"
#include "../../Organism/CPU/ObservationStuff.h"

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
    static void place_organisms (EngineDataContainer * dc, EngineControlParameters * cp);

    static void place_organism (EngineDataContainer * dc, Organism * organism);

    static void eat_food       (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void tick_lifetime(EngineDataContainer *dc, std::vector<std::vector<int>> &to_erase,
                              Organism *organism, int thread_num, int organism_pos);

    static void erase_organisms (EngineDataContainer * dc, std::vector<std::vector<int>>& to_erase);

    static void get_observations(EngineDataContainer *dc, Organism *&organism,
                                 std::vector<std::vector<Observation>> &organism_observations,
                                 SimulationParameters *sp, int organism_num);

    static void start_stage(EngineDataContainer *dc, PartialSimulationStage stage);

    static void change_organisms_pools(EngineDataContainer *dc, EngineControlParameters * cp);

    static void sort_organisms(EngineDataContainer *dc, EngineControlParameters *cp);

    static int calculate_num_organisms(EngineDataContainer *dc);

public:
    static void
    partial_multi_thread_tick(EngineDataContainer *dc, EngineControlParameters *cp,
                              OrganismBlockParameters *bp, SimulationParameters *sp,
                              lehmer64 *gen);

    static void build_threads(EngineDataContainer &dc, EngineControlParameters &cp,
                              SimulationParameters &sp);

    static void kill_threads(EngineDataContainer &dc);

    static void thread_tick(PartialSimulationStage stage, EngineDataContainer *dc,
                            SimulationParameters *sp, lehmer64 *gen, int thread_num);

    static void init(EngineDataContainer &dc, EngineControlParameters &cp,
                     SimulationParameters &sp);

    static void stop(EngineDataContainer &dc, EngineControlParameters &cp,
                     SimulationParameters &sp);
};

struct eager_worker_partial {
    EngineDataContainer * dc = nullptr;
    SimulationParameters * sp = nullptr;
    int thread_num;

    std::random_device r;
    lehmer64 gen;

    eager_worker_partial() = default;
    eager_worker_partial(EngineDataContainer * dc, SimulationParameters * sp, int thread_num):
            dc(dc), sp(sp), thread_num(thread_num){
        gen = lehmer64(r());
        thread = std::thread{&eager_worker_partial::main_loop, this};
    }
    eager_worker_partial(const eager_worker_partial & worker):
            dc(worker.dc), sp(worker.sp), thread_num(worker.thread_num){
        gen = lehmer64(r());
        thread = std::thread{&eager_worker_partial::main_loop, this};
    }

    inline void work(PartialSimulationStage stage) {
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
            SimulationEnginePartialMultiThread::thread_tick(thread_stage, dc, sp, &gen, thread_num);
            has_work = false;
        }
    }

    std::thread thread;
};


#endif //THELIFEENGINECPP_SIMULATIONENGINEPARTIALMULTITHREAD_H
