//
// Created by spaceeye on 30.03.2022.
//

#ifndef THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
#define THELIFEENGINECPP_ENGINECONTROLCONTAINER_H

enum class SimulationModes {
    CPU_Single_Threaded,
    CPU_Partial_Multi_threaded,
    CPU_Multi_Threaded,
    GPU_CUDA_mode,
    OPENCL_MODE,
    GPUFORT_MODE //?

};

struct EngineControlParameters {
    // if false then engine will stop
    bool engine_working = true;

    bool stop_engine = false;
    // a signal for engine to pause working when true to parse data
    bool engine_pause = false;
    // pauses the engine when true by user input
    bool engine_global_pause = false;
    // will do one tick and then return to being stopped.
    bool engine_pass_tick = false;

    bool synchronise_simulation_tick = false;
    // a signal for window process that engine is stopped, and window process can parse data from engine
    bool engine_paused = false;
    // for image creating purposes
    bool calculate_simulation_tick_delta_time = true;
    // if true, will build the threads
    bool build_threads = false;

    bool pause_processing_user_action = false;
    bool processing_user_actions = true;

    bool pause_button_pause = false;

    bool pass_tick = false;

    bool organisms_extinct = false;

    bool tb_paused = false;
    bool reset_with_chosen = false;

    //TODO change this
    SimulationModes simulation_mode = SimulationModes::CPU_Single_Threaded;
    SimulationModes change_to_mode = SimulationModes::CPU_Single_Threaded;
    bool change_simulation_mode = false;

    uint8_t num_threads = 2;

};

#endif //THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
