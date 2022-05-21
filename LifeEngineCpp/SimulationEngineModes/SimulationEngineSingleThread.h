//
// Created by spaceeye on 16.05.2022.
//

#ifndef THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
#define THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H


#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include "../GridBlocks/BaseGridBlock.h"
#include "../Organism/Organism.h"
#include "../BlockTypes.hpp"
#include "../EngineControlContainer.h"
#include "../EngineDataContainer.h"
#include "../OrganismBlockParameters.h"
#include "../Linspace.h"

class SimulationEngineSingleThread {

    static void produce_food    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism, std::mt19937 & mt);

    static void eat_food        (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void tick_lifetime   (EngineDataContainer * dc, std::vector<int>& to_erase, Organism *organism, int organism_pos);

    static void erase_organisms (EngineDataContainer * dc, std::vector<int>& to_erase, int i);

    static void apply_damage    (EngineDataContainer * dc, SimulationParameters * sp, Organism *organism);

    static void reserve_observations(std::vector<std::vector<Observation>> &observations,
                                     std::vector<Organism *> &organisms,
                                     SimulationParameters *sp);

    static void get_observations(EngineDataContainer *dc, SimulationParameters *sp,
                                 std::vector<Organism *> &organisms,
                                 std::vector<std::vector<Observation>> &organism_observations);

    static void rotate_organism (EngineDataContainer * dc, Organism *organism, BrainDecision decision);

    static void move_organism   (EngineDataContainer * dc, Organism *organism, BrainDecision decision);

    static void make_decision   (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Observation> &organism_observations);

    static void try_make_child  (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Organism *> &child_organisms, std::mt19937 *mt);

    static void make_child      (EngineDataContainer * dc, Organism *organism, std::mt19937 * mt);

    static void place_child     (EngineDataContainer *dc, SimulationParameters *sp, Organism *organism, std::vector<Organism *> &child_organisms, std::mt19937 *mt);

    static bool check_if_out_of_boundaries(EngineDataContainer *dc, int x, int y);
    static bool check_if_block_out_of_boundaries(EngineDataContainer *dc, Organism *organism,
                                                 BaseSerializedContainer &block, Rotation rotation);

public:
    static void single_threaded_tick(EngineDataContainer * dc,
                                     SimulationParameters * sp,
                                     std::mt19937 * mt);
};


#endif //THELIFEENGINECPP_SIMULATIONENGINESINGLETHREAD_H
