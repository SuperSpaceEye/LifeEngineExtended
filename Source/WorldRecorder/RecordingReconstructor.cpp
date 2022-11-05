//
// Created by spaceeye on 02.10.22.
//

#include "RecordingReconstructor.h"

void RecordingReconstructor::start_reconstruction(int width, int height) {
    this->width = width;
    this->height = height;

    rec_grid.resize(width * height);
}

void RecordingReconstructor::apply_transaction(Transaction &transaction) {
    if (transaction.reset) {
        apply_reset(transaction);
    } else if (transaction.starting_point) {
        apply_starting_point(transaction);
    } else {
        apply_normal(transaction);
    }
}

void RecordingReconstructor::apply_starting_point(Transaction &transaction) {
    rec_grid = std::vector<BaseGridBlock>(width*height, BaseGridBlock{BlockTypes::EmptyBlock});
    rec_orgs = std::vector<Organism>();
    recenter_to_imaginary_pos = transaction.recenter_to_imaginary_pos;

    apply_normal(transaction);
}

void RecordingReconstructor::apply_reset(Transaction & transaction) {
    rec_grid = std::vector<BaseGridBlock>(width*height, BaseGridBlock{BlockTypes::EmptyBlock});
    rec_orgs = std::vector<Organism>();
    recenter_to_imaginary_pos = transaction.recenter_to_imaginary_pos;
    transaction.dead_organisms.clear();

    apply_normal(transaction);
}


void RecordingReconstructor::apply_normal(Transaction &transaction) {
    apply_food_change(transaction);
    apply_recenter(transaction);
    apply_dead_organisms(transaction);
    apply_compressed_change(transaction);
    apply_move_change(transaction);
    apply_wall_change(transaction);
    apply_organism_change(transaction);
}

void RecordingReconstructor::apply_wall_change(Transaction &transaction) {
    for (auto & wc: transaction.wall_change) {
        rec_grid[wc.x + wc.y * width].type = wc.added ? BlockTypes::WallBlock : BlockTypes::EmptyBlock;
    }
}

void RecordingReconstructor::apply_move_change(Transaction &transaction) {
    for (auto & mc: transaction.move_change) {
        auto & o = rec_orgs[mc.vector_index];

        for (auto & b: o.anatomy._organism_blocks) {
            auto & wb = rec_grid[o.x + b.get_pos(o.rotation).x + (o.y + b.get_pos(o.rotation).y) * width];
            wb.type = BlockTypes::EmptyBlock;
        }

        o.rotation = mc.rotation;
        o.x = mc.x;
        o.y = mc.y;

        for (auto & b: o.anatomy._organism_blocks) {
            auto & wb = rec_grid[o.x + b.get_pos(o.rotation).x + (o.y + b.get_pos(o.rotation).y) * width];
            wb.type = b.type;
            wb.rotation = b.rotation;
        }
    }
}

void RecordingReconstructor::apply_dead_organisms(Transaction &transaction) {
    for (auto & dc: transaction.dead_organisms) {
        auto & o = rec_orgs[dc];
        for (auto & b: o.anatomy._organism_blocks) {
            auto & wb = rec_grid[o.x + b.get_pos(o.rotation).x + (o.y + b.get_pos(o.rotation).y) * width];
            wb.type = BlockTypes::FoodBlock;
        }
    }
}

void RecordingReconstructor::apply_recenter(const Transaction &transaction) {
    if (recenter_to_imaginary_pos != transaction.recenter_to_imaginary_pos) {
        for (auto & organism: rec_orgs) {
            auto vec = organism.anatomy.recenter_blocks(transaction.recenter_to_imaginary_pos);

            auto temp = BaseSerializedContainer{vec.x, vec.y};
            organism.x += temp.get_pos(organism.rotation).x;
            organism.y += temp.get_pos(organism.rotation).y;
        }
        recenter_to_imaginary_pos = transaction.recenter_to_imaginary_pos;
    }
}

void RecordingReconstructor::apply_food_change(Transaction &transaction) {
    for (auto & fc: transaction.food_change) {
        rec_grid[fc.x + fc.y * width].type = fc.added ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
    }
}

void RecordingReconstructor::apply_organism_change(Transaction &transaction) {
    //TODO did it wrong
    rec_orgs.resize(rec_orgs.size() + transaction.organism_change.size());
    for (auto & o: transaction.organism_change) {
        for (auto & b: o.anatomy._organism_blocks) {
            auto & wb = rec_grid[o.x + b.get_pos(o.rotation).x + (o.y + b.get_pos(o.rotation).y) * width];
            wb.type = b.type;
            wb.rotation = b.rotation;
        }
        auto temp = o.vector_index;
        rec_orgs[temp] = o;
        rec_orgs[temp].vector_index = temp;
    }
}

void RecordingReconstructor::apply_compressed_change(Transaction &transaction) {
    for (auto & pair: transaction.compressed_change) {
        auto & left_organism  = rec_orgs[pair.first];
        auto & right_organism = rec_orgs[pair.second];

        auto left_index  = left_organism.vector_index;
        auto right_index = right_organism.vector_index;

        std::swap(left_organism, right_organism);

        left_organism.vector_index  = left_index;
        right_organism.vector_index = right_index;
    }
}

const std::vector<BaseGridBlock> &RecordingReconstructor::get_state() {
    return rec_grid;
}

void RecordingReconstructor::finish_reconstruction() {
    rec_grid = std::vector<BaseGridBlock>();
    rec_orgs = std::vector<Organism>();
    width = 0;
    height = 0;
}
