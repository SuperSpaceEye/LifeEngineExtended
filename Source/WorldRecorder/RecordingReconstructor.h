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

    bool initialized = false;
    bool recenter_to_imaginary_pos;

    std::vector<BaseGridBlock> rec_grid;
    std::vector<Organism> rec_orgs;

    void apply_organism_change(Transaction &transaction);
    void apply_food_change(Transaction &transaction);
    void apply_recenter(const Transaction &transaction);
    void apply_dead_organisms(Transaction &transaction);
    void apply_move_change(Transaction &transaction);
    void apply_wall_change(Transaction &transaction);
public:
    RecordingReconstructor()=default;

    void start_reconstruction(int width, int height);

    void apply_transaction(Transaction & transaction);

    void apply_starting_point(Transaction & transaction);
    void apply_reset(Transaction & transaction);
    void apply_normal(Transaction & transaction);

    const std::vector<BaseGridBlock> & get_state();

    void finish_reconstruction();
};


#endif //LIFEENGINEEXTENDED_RECORDINGRECONSTRUCTOR_H
