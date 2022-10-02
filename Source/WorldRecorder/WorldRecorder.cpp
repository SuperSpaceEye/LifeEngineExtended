//
// Created by spaceeye on 02.10.22.
//

#include "WorldRecorder.h"

#include <utility>

const int BUFFER_VERSION = 1;

void TransactionBuffer::start_recording(std::string path_to_save_, int width_, int height_, int buffer_size_) {
    path_to_save = std::move(path_to_save_);
    buffer_size = buffer_size_;
    width = width_;
    height = height_;
    buffer_pos = 0;
    saved_buffers = 1;
    recorded_transactions = 0;

    transactions.reserve(buffer_size);
    transactions.emplace_back();
    transactions[0].starting_point = true;
}

void TransactionBuffer::record_food_change(int x, int y, bool added) {
    transactions[buffer_pos].food_change.emplace_back(x, y, added);
}

void TransactionBuffer::record_new_organism(Organism &organism) {
    auto new_organism = Organism(&organism);
    new_organism.vector_index = organism.vector_index;
    transactions[buffer_pos].organism_change.push_back(OrganismChange{std::move(new_organism)});
}

void TransactionBuffer::record_organism_dying(int organism_index) {
    transactions[buffer_pos].dead_organisms.emplace_back(organism_index);
}

void TransactionBuffer::record_organism_move_change(int vector_index, int x, int y, Rotation rotation) {
    transactions[buffer_pos].move_change.emplace_back(vector_index, rotation, x, y);
}

void TransactionBuffer::record_recenter_to_imaginary_pos(bool state) {
    transactions[buffer_pos].recenter_to_imaginary_pos = state;
}

void TransactionBuffer::record_wall_changes(int x, int y, bool added) {
    transactions[buffer_pos].wall_change.emplace_back(x, y, added);
}

void TransactionBuffer::record_reset() {
    transactions[buffer_pos].reset = true;
}

void TransactionBuffer::record_transaction() {
    if (transactions.size() < buffer_size) {
        transactions.emplace_back();
        buffer_pos++;
        recorded_transactions++;
        transactions[buffer_pos].starting_point = false;
        return;
    }

    flush_transactions();
    transactions.clear();
    transactions.emplace_back();
    buffer_pos = 0;
}

void TransactionBuffer::flush_transactions() {
    if (buffer_pos == 0) { return;}

    auto path = path_to_save + std::to_string(saved_buffers);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write((char*)&BUFFER_VERSION, sizeof(int));
    out.write((char*)&width, sizeof(int));
    out.write((char*)&height, sizeof(int));
    out.write((char*)&buffer_pos, sizeof(int));

    for (auto & transaction: transactions) {
        out.write((char*)&transaction.starting_point, sizeof(bool));
        out.write((char*)&transaction.recenter_to_imaginary_pos, sizeof(bool));

        int size = transaction.organism_change.size();
        out.write((char*)&size, sizeof(int));
        out.write((char*)transaction.organism_change.data(), sizeof(OrganismChange)*size);

        size = transaction.food_change.size();
        out.write((char*)&size, sizeof(int));
        out.write((char*)transaction.food_change.data(), sizeof(FoodChange)*size);

        size = transaction.dead_organisms.size();
        out.write((char*)&size, sizeof(int));
        out.write((char*)transaction.dead_organisms.data(), sizeof(int)*size);

        size = transaction.move_change.size();
        out.write((char*)&size, sizeof(int));
        out.write((char*)transaction.move_change.data(), sizeof(MoveChange)*size);

        size = transaction.wall_change.size();
        out.write((char*)&size, sizeof(int));
        out.write((char*)transaction.wall_change.data(), sizeof(WallChange)*size);
    }

    out.close();

    saved_buffers++;
}

void TransactionBuffer::finish_recording() {
    flush_transactions();

    width = 0;
    height = 0;
    buffer_pos = 0;
    saved_buffers = 1;
    buffer_size = 0;
    path_to_save = "";
    recorded_transactions = 0;
    transactions = std::vector<Transaction>();
}