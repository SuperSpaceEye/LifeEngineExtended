//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINEDATACONTAINER_H
#define THELIFEENGINECPP_ENGINEDATACONTAINER_H

//TODO restructure elements in a struct
struct EngineDataContainer {
    // dimensions of the simulation
    int simulation_width = 600;
    int simulation_height = 600;
    int engine_ticks = 0;
    long delta_time;
    float simulation_interval = 0.;
    bool unlimited_simulation_fps = true;
    std::vector<std::vector<BaseGridBlock>> simulation_grid;
    std::vector<std::vector<BaseGridBlock>> second_simulation_grid;
    std::vector<Organism> organisms;
    // adding/killing organisms, adding/deleting food/walls, etc.
    //std::vector<Action> user_actions_pool;;

};

#endif //THELIFEENGINECPP_ENGINEDATACONTAINER_H
