// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 02.10.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTOR_H
#define LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTOR_H

#include "WorldRecorder.h"

class RecordingReconstructor {
private:
    int width;
    int height;

    bool recenter_to_imaginary_pos;
    float food_threshold;

    std::vector<BaseGridBlock> rec_grid;
    std::vector<float> food_grid;
    std::vector<Organism> rec_orgs;

    void apply_organism_change(WorldRecorder::Transaction &transaction);
    void apply_food_change(WorldRecorder::Transaction &transaction);
    void apply_recenter(const WorldRecorder::Transaction &transaction);
    void apply_food_threshold(WorldRecorder::Transaction & transaction);
    void apply_dead_organisms(WorldRecorder::Transaction &transaction);
    void apply_move_change(WorldRecorder::Transaction &transaction);
    void apply_compressed_change(WorldRecorder::Transaction &transaction);

    void apply_user_wall_change(WorldRecorder::Transaction &transaction, int pos);
    void apply_user_food_change(WorldRecorder::Transaction &transaction, int pos);
    void apply_user_add_organism(WorldRecorder::Transaction &transaction, int pos);
    void apply_user_kill_organism(WorldRecorder::Transaction &transaction, int pos);

    void apply_user_actions(WorldRecorder::Transaction &transaction);

    void apply_starting_point(WorldRecorder::Transaction & transaction);
    void apply_reset(WorldRecorder::Transaction & transaction);
    void apply_normal(WorldRecorder::Transaction & transaction);
    //TODO void_apply_user_food_change
public:
    RecordingReconstructor()=default;

    void start_reconstruction(int width, int height);

    void apply_transaction(WorldRecorder::Transaction & transaction);

    const std::vector<BaseGridBlock> & get_state();

    void finish_reconstruction();
};


#endif //LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTOR_H
