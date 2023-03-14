// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 02.10.22.
//

#include "RecordingReconstructor.h"
#include "WorldRecorder.h"

void RecordingReconstructor::start_reconstruction(int width, int height) {
    this->width = width;
    this->height = height;

    rec_grid.resize(width * height);
}

void RecordingReconstructor::apply_transaction(WorldRecorder::Transaction &transaction) {
    if (transaction.reset) {
        apply_reset(transaction);
    } else if (transaction.starting_point) {
        apply_starting_point(transaction);
    } else {
        apply_normal(transaction);
    }
}

void RecordingReconstructor::apply_starting_point(WorldRecorder::Transaction &transaction) {
    rec_grid = std::vector<BaseGridBlock>(width*height, BaseGridBlock{BlockTypes::EmptyBlock});
    food_grid = std::vector<float>(width*height, 0);
    rec_orgs = std::vector<Organism>();
    recenter_to_imaginary_pos = transaction.recenter_to_imaginary_pos;

    apply_organism_change(transaction);
    transaction.organism_change.clear();

    apply_normal(transaction);
}

void RecordingReconstructor::apply_reset(WorldRecorder::Transaction & transaction) {
    rec_grid = std::vector<BaseGridBlock>(width*height, BaseGridBlock{BlockTypes::EmptyBlock});
    food_grid = std::vector<float>(width*height, 0);
    rec_orgs = std::vector<Organism>();
    recenter_to_imaginary_pos = transaction.recenter_to_imaginary_pos;
    transaction.dead_organisms.clear();

    apply_normal(transaction);
}


void RecordingReconstructor::apply_normal(WorldRecorder::Transaction &transaction) {
    apply_user_actions(transaction);
    apply_recenter(transaction);
    apply_food_threshold(transaction);

    apply_organism_size_change(transaction);
    apply_food_change(transaction);
    apply_dead_organisms(transaction);
    apply_move_change(transaction);
    apply_compressed_change(transaction);
    apply_organism_change(transaction);
}

void RecordingReconstructor::apply_user_actions(WorldRecorder::Transaction &transaction) {
    int wall_change_pos = 0;
    int food_change_pos = 0;
    int organism_change_pos = 0;
    int organism_kill_pos = 0;

    for (auto type: transaction.user_action_execution_order) {
        switch (type) {
            case WorldRecorder::RecActionType::WallChange:     apply_user_wall_change(transaction, wall_change_pos++); break;
            case WorldRecorder::RecActionType::FoodChange:     apply_user_food_change(transaction, food_change_pos++);break;
            case WorldRecorder::RecActionType::OrganismChange: apply_user_add_organism(transaction, organism_change_pos++); break;
            case WorldRecorder::RecActionType::OrganismKill:   apply_user_kill_organism(transaction, organism_kill_pos++); break;
        }
    }
}

void RecordingReconstructor::apply_user_wall_change(WorldRecorder::Transaction &transaction, int pos) {
    auto & wc = transaction.user_wall_change[pos];
    rec_grid[wc.x + wc.y * width].type = wc.added ? BlockTypes::WallBlock : BlockTypes::EmptyBlock;
}

void RecordingReconstructor::apply_user_kill_organism(WorldRecorder::Transaction &transaction, int pos) {
    auto dc = transaction.user_dead_change[pos];
    auto & o = rec_orgs[dc];
    for (auto & b: o.anatomy.organism_blocks) {
        const auto bpos = b.get_pos(o.rotation);
        const auto apos = o.x + bpos.x + (o.y + bpos.y) * width;

        auto & wb = rec_grid[apos];
        wb.type = food_grid[apos] >= food_threshold ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
    }
}

void RecordingReconstructor::apply_user_food_change(WorldRecorder::Transaction &transaction, int pos) {
    const auto & fc = transaction.user_food_change[pos];
    const auto apos = fc.x + fc.y * width;

    auto & num = food_grid[apos];
    auto & type = rec_grid[apos].type;
    num += fc.num;

    if (type == BlockTypes::EmptyBlock || type == BlockTypes::FoodBlock) {
        type = num > food_threshold ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
    }
}

void RecordingReconstructor::apply_user_add_organism(WorldRecorder::Transaction &transaction, int pos) {
    auto & o = transaction.user_organism_change[pos];
    for (auto & b: o.anatomy.organism_blocks) {
        const auto bpos = b.get_pos(o.rotation);
        auto & wb = rec_grid[o.x + bpos.x + (o.y + bpos.y) * width];
        wb.type = b.type;
        wb.rotation = b.rotation;
    }
    auto temp = o.vector_index;
    if (temp+1 > rec_orgs.size()) {rec_orgs.resize(temp+1);}
    rec_orgs[temp] = o;
    rec_orgs[temp].vector_index = temp;
}

void RecordingReconstructor::apply_organism_size_change(WorldRecorder::Transaction &transaction) {
    for (auto & sc: transaction.organism_size_change) {
        rec_orgs[sc.organism_idx].size = sc.new_size;
    }
}

void RecordingReconstructor::apply_move_change(WorldRecorder::Transaction &transaction) {
    for (auto & mc: transaction.move_change) {
        auto & o = rec_orgs[mc.vector_index];

        for (auto & b: o.get_organism_blocks_view()) {
            const auto bpos = b.get_pos(o.rotation);
            const auto apos = o.x + bpos.x + (o.y + bpos.y) * width;

            auto & wb = rec_grid[apos];
            if (food_grid[apos] >= food_threshold) {
                wb.type = BlockTypes::FoodBlock;
            } else {
                wb.type = BlockTypes::EmptyBlock;
            }
        }

        o.rotation = mc.rotation;
        o.x = mc.x;
        o.y = mc.y;

        for (auto & b: o.get_organism_blocks_view()) {
            const auto bpos = b.get_pos(o.rotation);
            auto & wb = rec_grid[o.x + bpos.x + (o.y + bpos.y) * width];
            wb.type = b.type;
            wb.rotation = b.rotation;
        }
    }
}

void RecordingReconstructor::apply_dead_organisms(WorldRecorder::Transaction &transaction) {
    for (auto & dc: transaction.dead_organisms) {
        auto & o = rec_orgs[dc];
        for (auto & b: o.get_organism_blocks_view()) {
            const auto bpos = b.get_pos(o.rotation);
            const auto apos = o.x + bpos.x + (o.y + bpos.y) * width;

            auto & wb = rec_grid[apos];
            wb.type = food_grid[apos] >= food_threshold ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
        }
    }
}

void RecordingReconstructor::apply_recenter(const WorldRecorder::Transaction &transaction) {
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

void RecordingReconstructor::apply_food_threshold(WorldRecorder::Transaction &transaction) {
    food_threshold = transaction.food_threshold;
}

void RecordingReconstructor::apply_food_change(WorldRecorder::Transaction &transaction) {
    for (auto & fc: transaction.food_change) {
        const auto apos = fc.x + fc.y * width;
        auto & num = food_grid[apos];
        auto & type = rec_grid[apos].type;
        num += fc.num;

        //TODO
        if (type == BlockTypes::EmptyBlock || type == BlockTypes::FoodBlock) {
            type = num >= 1 ? BlockTypes::FoodBlock : BlockTypes::EmptyBlock;
        }
    }
}

void RecordingReconstructor::apply_organism_change(WorldRecorder::Transaction &transaction) {
    //TODO did resizing wrong
    rec_orgs.resize(rec_orgs.size() + transaction.organism_change.size());
    for (auto & o: transaction.organism_change) {
        for (auto & b: o.get_organism_blocks_view()) {
            auto & wb = rec_grid[o.x + b.get_pos(o.rotation).x + (o.y + b.get_pos(o.rotation).y) * width];
            wb.type = b.type;
            wb.rotation = b.rotation;
        }
        auto temp = o.vector_index;
        if (temp+1 > rec_orgs.size()) {rec_orgs.resize(temp+1);}
        rec_orgs[temp] = o;
        rec_orgs[temp].vector_index = temp;
    }
}

void RecordingReconstructor::apply_compressed_change(WorldRecorder::Transaction &transaction) {
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
    food_grid = std::vector<float>();
    width = 0;
    height = 0;
}
