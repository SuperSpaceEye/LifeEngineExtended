// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 02.10.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDRECORDER_H
#define LIFEENGINEEXTENDED_WORLDRECORDER_H

#include <vector>
#include <cctype>
#include <fstream>

#include "Organism/Organism.h"

namespace WorldRecorder {
struct FoodChange {
    int x;
    int y;
    float num;

    FoodChange()=default;
    FoodChange(int x, int y, float num): x(x), y(y), num(num){}
};

struct WallChange {
    int x;
    int y;
    bool added;

    WallChange()=default;
    WallChange(int x, int y, bool added): x(x), y(y), added(added){}
};

struct MoveChange {
    int vector_index;
    Rotation rotation;
    int x;
    int y;

    MoveChange()=default;
    MoveChange(int vector_index, Rotation rotation, int x, int y): vector_index(vector_index), rotation(rotation), x(x), y(y){}
};

struct OSizeChange {
    int new_size;
    uint32_t organism_idx;

    OSizeChange()=default;
    OSizeChange(int new_size, uint32_t organism_idx): new_size(new_size), organism_idx(organism_idx) {}
};

//TODO recenter
enum class RecActionType {
    WallChange,
    FoodChange,
    OrganismChange,
    OrganismKill
};
struct Transaction {
    std::vector<Organism> organism_change;
    std::vector<FoodChange> food_change;
    std::vector<int> dead_organisms;
    std::vector<MoveChange> move_change;
    std::vector<std::pair<int, int>> compressed_change;
    std::vector<OSizeChange> organism_size_change;

    std::vector<RecActionType> user_action_execution_order;
    std::vector<WallChange> user_wall_change;
    std::vector<FoodChange> user_food_change;
    std::vector<Organism> user_organism_change;
    std::vector<int> user_dead_change;
    float food_threshold;
    bool starting_point;
    bool recenter_to_imaginary_pos;
    bool reset = false;
    bool uses_occ = false;
};

struct TransactionBuffer {
    std::vector<Transaction> transactions;
    std::string path_to_save;
    int buffer_size = 5000;
    int buffer_pos = 0;
    int recorded_transactions = 0;
    int saved_buffers = 1;
    int width;
    int height;

    void start_recording(std::string path_to_save, int width, int height, int buffer_size = 2000);

    void record_food_change(int x, int y, float num);
    void record_new_organism(const Organism &organism);
    void record_organism_dying(int organism_index);
    void record_organism_move_change(int vector_index, int x, int y, Rotation rotation);
    void record_recenter_to_imaginary_pos(bool state);
    void record_food_threshold(float food_threshold);
    void record_reset();
    void record_compressed(int pos1, int pos2);
    void record_organism_size_change(const Organism &organism);

    void record_user_wall_change(int x, int y, bool added);
    void record_user_food_change(int x, int y, float num);
    void record_user_new_organism(const Organism &organism);
    void record_user_kill_organism(int organism_index);

    void resize_buffer(int new_buffer_size);
    void flush_transactions();
    static bool load_buffer_metadata(std::string & path_to_buffer, int & width, int & height, int & piece_len);
    bool load_buffer(std::string & path_to_buffer);
    void record_transaction();
    void finish_recording();
};
}


#endif //LIFEENGINEEXTENDED_WORLDRECORDER_H
