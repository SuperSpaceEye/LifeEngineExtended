//
// Created by spaceeye on 02.10.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDRECORDER_H
#define LIFEENGINEEXTENDED_WORLDRECORDER_H

#include <vector>
#include <cctype>
#include <fstream>

#include "../Organism/CPU/Organism.h"

struct FoodChange {
    int x;
    int y;
    bool added;

    FoodChange()=default;
    FoodChange(int x, int y, bool added): x(x), y(y), added(added){}
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

struct Transaction {
    std::vector<Organism> organism_change;
    std::vector<FoodChange> food_change;
    std::vector<int> dead_organisms;
    std::vector<MoveChange> move_change;
    std::vector<WallChange> wall_change;
    bool starting_point;
    bool recenter_to_imaginary_pos;
    bool reset = false;
    bool uses_occ = false;
};

struct TransactionBuffer {
    std::vector<Transaction> transactions;
    std::string path_to_save;
    int buffer_size = 2000;
    int buffer_pos = 0;
    int recorded_transactions = 0;
    int saved_buffers = 1;
    int width;
    int height;

    void start_recording(std::string path_to_save, int width, int height, int buffer_size = 2000);

    void record_food_change(int x, int y, bool added);
    void record_new_organism(Organism & organism);
    void record_organism_dying(int organism_index);
    void record_organism_move_change(int vector_index, int x, int y, Rotation rotation);
    void record_recenter_to_imaginary_pos(bool state);
    void record_wall_changes(int x, int y, bool added);
    void record_reset();

    void resize_buffer(int new_buffer_size);
    void flush_transactions();
    static bool load_buffer_metadata(std::string & path_to_buffer, int & width, int & height, int & piece_len);
    bool load_buffer(std::string & path_to_buffer);
    void record_transaction();
    void finish_recording();
};


#endif //LIFEENGINEEXTENDED_WORLDRECORDER_H
