// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

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
//    OPENCL_MODE,
//    GPUFORT_MODE //?

};

struct EngineControlParameters {
    // if false then engine will y
    volatile bool engine_working = true;

    volatile bool stop_engine = false;
    // a signal for engine to pause working when true to parse data
    volatile bool engine_pause = false;
    // pauses the engine when true by user input
    volatile bool engine_global_pause = false;
    // will do one tick and then return to being stopped.
    volatile bool engine_pass_tick = false;

    volatile bool synchronise_simulation_tick = false;
    // a signal for window process that engine is stopped, and window process can parse data from engine
    volatile bool engine_paused = false;
    // for image creating purposes
    volatile bool calculate_simulation_tick_delta_time = true;
    // if true, will build the threads
    volatile bool build_threads = false;

    volatile bool pause_processing_user_action = false;
    volatile bool processing_user_actions = true;

    volatile bool pause_button_pause = false;

    volatile bool pass_tick = false;

    volatile bool organisms_extinct = false;

    volatile bool tb_paused = false;
    volatile bool reset_with_editor_organism = false;

    SimulationModes simulation_mode = SimulationModes::CPU_Single_Threaded;
    SimulationModes change_to_mode = SimulationModes::CPU_Single_Threaded;
    volatile bool change_simulation_mode = false;

    volatile bool update_editor_organism = false;

    uint8_t num_threads = 2;

};

#endif //THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
