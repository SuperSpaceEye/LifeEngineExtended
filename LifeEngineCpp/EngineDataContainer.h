//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

struct EngineDataContainer {
    uint64_t delta_time = 0;
    uint32_t engine_ticks = 0;
    // dimensions of the simulation
    uint16_t simulation_width = 600;
    uint16_t simulation_height = 600;
    float simulation_interval = 0.;
    bool unlimited_simulation_fps = true;
    std::vector<std::vector<BaseGridBlock>> simulation_grid;
    std::vector<std::vector<BaseGridBlock>> second_simulation_grid;
    std::vector<Organism> organisms;
    // adding/killing organisms, adding/deleting food/walls, etc.
    //std::vector<Action> user_actions_pool;;

};

#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
