//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
#define THELIFEENGINECPP_ENGINECONTROLCONTAINER_H

struct EngineControlParameters {
    // if false then engine will stop
    bool engine_working = true;
    // a signal for engine to pause working when true to parse data
    bool engine_pause = false;
    // pauses the engine when true by user input
    bool engine_global_pause = false;
    // will do one tick and then return to being stopped.
    bool engine_pass_tick = false;
    // a signal for window process that engine is stopped, and window process can parse data from engine
    bool engine_paused = false;
    // for image creating purposes
    bool calculate_simulation_tick_delta_time = true;
};

#endif //THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
