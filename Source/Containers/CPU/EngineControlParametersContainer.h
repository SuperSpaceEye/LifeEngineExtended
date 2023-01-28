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

//should be only for communication between main and simulation threads.
struct EngineControlParameters {
    volatile bool engine_working = true;
    volatile bool stop_engine = false;

    // a signal for engine to pause working when true
    volatile bool engine_pause = false;
    // a signal for main process to know when engine is stopped
    volatile bool engine_paused = false;
    // will do one tick and then return to being stopped.
    volatile bool engine_pass_tick = false;
    // getting time is expensive, so if simulation time set to unlimited, do not get time.
    volatile bool calculate_simulation_tick_delta_time = true;
    //TODO move into sim engine?
    volatile bool organisms_extinct = false;

    // pauses the engine when ui pause button is pressed
    volatile bool engine_global_pause = false;
    // only for pass tick button.
    volatile bool pass_tick = false;

    //are not used right now
    volatile bool build_threads = false;
    volatile bool change_simulation_mode = false;
    volatile uint8_t num_threads = 2;
    SimulationModes simulation_mode = SimulationModes::CPU_Single_Threaded;
    SimulationModes change_to_mode = SimulationModes::CPU_Single_Threaded;
    //

    volatile bool tb_paused = false;
    volatile bool reset_with_editor_organism = false;
    volatile bool pause_button_pause = false;

    volatile bool update_editor_organism = false;

    //TODO move to recorder?
    volatile int  parse_full_grid_every_n = 1;

    //TODO move?
    volatile bool lock_resizing = false;

    volatile int  update_info_every_n_tick = 100;
    volatile int  update_world_events_every_n_tick = 1;
    volatile bool execute_world_events = false;
    volatile bool pause_world_events = false;
    volatile bool update_world_events_ui_once = false;

    volatile bool do_not_use_user_actions_ui = false;
    volatile bool do_not_use_user_actions_engine = false;
};

#endif //THELIFEENGINECPP_ENGINECONTROLCONTAINER_H
