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
#include "../Organism/Organism.h"
#include "../BlockTypes.hpp"
#include "../EngineControlContainer.h"
#include "../EngineDataContainer.h"
#include "../OrganismBlockParameters.h"
#include "../Linspace.h"

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
    MakeDecision,
    //DoDecision,
    //TryMakeChild
};

class SimulationEnginePartialMultiThread {
private:
    static void place_organisms (EngineDataContainer * dc);

    static void produce_food    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, std::mt19937 & mt);

    static void eat_food        (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void tick_lifetime   (EngineDataContainer * dc, std::vector<int>& to_erase, Organism *organism, int organism_pos);

    static void erase_organisms (EngineDataContainer * dc, std::vector<int>& to_erase, int i);

    static void apply_damage    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void reserve_observations(std::vector<std::vector<Observation>> & observations, std::vector<Organism *> &organisms);

    static void get_observations(EngineDataContainer *dc, std::vector<Organism *> &organisms, std::vector<std::vector<Observation>> &organism_observations);

    static void rotate_organism (EngineDataContainer * dc, Organism *organism, BrainDecision decision);

    static void move_organism   (EngineDataContainer * dc, Organism *organism, BrainDecision decision);

    static void make_decision   (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Observation> &organism_observations);

    static void do_decision   (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Observation> &organism_observations);

    static void try_make_child  (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Organism *> &child_organisms, std::mt19937 *mt);

    static void make_child      (EngineDataContainer * dc, Organism *organism, std::mt19937 * mt);

    static void place_child     (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Organism *> &child_organisms, std::mt19937 *mt);

    static bool check_if_out_of_boundaries(EngineDataContainer *dc, Organism *organism,
                                           BaseSerializedContainer &block, Rotation rotation);

    static void calculate_threads_points(int num_organisms, int num_threads, std::vector<int> & thread_points);

public:
    static void
    partial_multi_thread_tick(EngineDataContainer *dc, EngineControlParameters *cp, OrganismBlockParameters *bp,
                              SimulationParameters *sp);

    static void build_threads(EngineDataContainer &dc, EngineControlParameters &cp);

    static void kill_threads(EngineDataContainer &dc);

    static void thread_tick(PartialSimulationStage stage, int start_pos, int end_pos);

    static void start_stage(PartialSimulationStage stage, std::vector<int> & thread_points);
};

struct eager_worker_partial {
    EngineDataContainer * dc = nullptr;
    int start_organism_pos = 0;
    int end_organism_pos = 0;

    std::random_device r;
    std::mt19937 mt;

    eager_worker_partial() = default;
    eager_worker_partial(EngineDataContainer * dc):
            dc(dc){
        auto seed = std::seed_seq{r(),r(),r(),r(),r(),r(),r(),r()};
        mt = std::mt19937(seed);
    }
    eager_worker_partial(const eager_worker_partial & worker):
            dc(worker.dc){
        auto seed = std::seed_seq{r(),r(),r(),r(),r(),r(),r(),r()};
        mt = std::mt19937(seed);
    }

    inline void work(PartialSimulationStage stage) {
        has_work.store(true);
        thread_stage.store(stage);
    }

    inline void finish() {
        while (has_work.load()) {}
    }

    inline void stop_work() {
        exiting.store(true);
    }

    inline ~eager_worker_partial() { stop_thread(); }
    inline void stop_thread() {
        exiting.store(true);
        has_work.store(true);
        if (thread.joinable()) {
            thread.join();
        }
    }
private:
    std::atomic<bool> has_work{false};

    std::atomic<bool> exiting{false};
    std::atomic<bool> thread_started{false};

    std::atomic<PartialSimulationStage> thread_stage;

    std::thread thread = std::thread([this] {
        thread_started.store(true);
        while (true) {
            while (!has_work.load()) {
                if (exiting.load()) {
                    return;
                }
            }
            SimulationEnginePartialMultiThread::thread_tick(thread_stage, 0, 0);
            has_work.store(false);
        }
    });
};


#endif //THELIFEENGINECPP_SIMULATIONENGINEPARTIALMULTITHREAD_H
