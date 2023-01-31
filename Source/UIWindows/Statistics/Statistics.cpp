//
// Created by spaceeye on 31.01.23.
//

#include "Statistics.h"
#include "../../Stuff/MiscFuncs.h"

void Statistics::make_organism_blocks_labels() {
    std::array<QVBoxLayout*, 3> layouts{ui.both_vl, ui.stationary_vl, ui.moving_vl};
    for (int n = 0; n < 3; n++) {
        for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
            auto lb = new QLabel(this);
            labels[n][i] = lb;
            layouts[n]->addWidget(lb);
        }
    }
}

void Statistics::update_statistics(const OrganismInfoContainer &info, EngineDataContainer & edc, int float_precision, float scaling_zoom, float center_x, float center_y) {
    ui.lb_total_engine_ticks ->setText(QString::fromStdString("Total engine ticks: " + std::to_string(edc.total_engine_ticks)));
    ui.lb_organisms_memory_consumption->setText(QString::fromStdString("Organisms memory consumption: " + convert_num_bytes(info.total_size)));
    ui.lb_organisms_alive_2    ->setText(QString::fromStdString("Organism alive: " + std::to_string(info.avgs[0].total)));
    ui.lb_organism_size_4      ->setText(QString::fromStdString("Avg organism size: " + to_str(info.avgs[0].size, float_precision)));
    ui.lb_avg_org_lifetime_4   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.avgs[0]._organism_lifetime, float_precision)));
    ui.lb_avg_gathered_food_4  ->setText(QString::fromStdString("Avg gathered food: " + to_str(info.avgs[0]._gathered_food, float_precision)));
    ui.lb_avg_age_4            ->setText(QString::fromStdString("Avg organism age: " + to_str(info.avgs[0]._organism_age, float_precision)));
    ui.lb_anatomy_mutation_rate_4 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.avgs[0].anatomy_mutation_rate, float_precision)));
    ui.lb_brain_mutation_rate_4   ->setText(QString::fromStdString("Avg brain mutation rate: " + to_str(info.avgs[0].brain_mutation_rate, float_precision)));
    ui.lb_avg_occ_length_4     ->setText(QString::fromStdString("Avg OCC length: " + to_str(info.avgs[0].occ_instructions_num)));
    ui.lb_total_occ_length_4   ->setText(QString::fromStdString("Total OCC length: " + std::to_string(info.avgs[0].total_occ_instructions_num)));


    ui.lb_moving_organisms     ->setText(QString::fromStdString("Moving organisms: " + std::to_string(info.avgs[2].total)));
    ui.lb_organisms_with_eyes  ->setText(QString::fromStdString("Organisms with eyes: " + std::to_string(info.organisms_with_eyes)));
    ui.lb_avg_org_lifetime_2   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.avgs[2]._organism_lifetime, float_precision)));
    ui.lb_avg_gathered_food_2  ->setText(QString::fromStdString("Avg gathered food: " + to_str(info.avgs[2]._gathered_food, float_precision)));
    ui.lb_avg_age_2            ->setText(QString::fromStdString("Avg organism age: " + to_str(info.avgs[2]._organism_age, float_precision)));
    ui.lb_average_moving_range ->setText(QString::fromStdString("Avg moving range: " + to_str(info.move_range, float_precision)));
    ui.lb_organism_size_2      ->setText(QString::fromStdString("Avg organism size: " + to_str(info.avgs[2].size, float_precision)));
    ui.lb_anatomy_mutation_rate_2 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.avgs[2].anatomy_mutation_rate, float_precision)));
    ui.lb_brain_mutation_rate_2   ->setText(QString::fromStdString("Avg brain mutation rate: " + to_str(info.avgs[2].brain_mutation_rate, float_precision)));
    ui.lb_avg_occ_len_2        ->setText(QString::fromStdString("Avg OCC length: " + to_str(info.avgs[2].occ_instructions_num)));
    ui.lb_total_occ_len_2      ->setText(QString::fromStdString("Total OCC length: " + std::to_string(info.avgs[2].total_occ_instructions_num)));


    ui.lb_stationary_organisms ->setText(QString::fromStdString("Stationary organisms: " + std::to_string(info.avgs[1].total)));
    ui.lb_organism_size_3      ->setText(QString::fromStdString("Avg organism size: " + to_str(info.avgs[1].size, float_precision)));
    ui.lb_avg_org_lifetime_3   ->setText(QString::fromStdString("Avg organism lifetime: " + to_str(info.avgs[1]._organism_lifetime, float_precision)));
    ui.lb_avg_gathered_food_3  ->setText(QString::fromStdString("Avg gathered food: " + to_str(info.avgs[1]._gathered_food, float_precision)));
    ui.lb_avg_age_3            ->setText(QString::fromStdString("Avg organism age: " + to_str(info.avgs[1]._organism_age, float_precision)));
    ui.lb_anatomy_mutation_rate_3 ->setText(QString::fromStdString("Avg anatomy mutation rate: " + to_str(info.avgs[1].anatomy_mutation_rate, float_precision)));
    ui.lb_brain_mutation_rate_3   ->setText(QString::fromStdString("Avg brain mutation rate: " + to_str(info.avgs[1].brain_mutation_rate, float_precision)));
    ui.lb_avg_occ_len_3        ->setText(QString::fromStdString("Avg OCC length: " + to_str(info.avgs[1].occ_instructions_num)));
    ui.lb_total_occ_length_3   ->setText(QString::fromStdString("Total OCC length: " + std::to_string(info.avgs[1].total_occ_instructions_num)));


    ui.lb_child_organisms         ->setText(QString::fromStdString("Child organisms: " + std::to_string(edc.stc.child_organisms.size())));
    ui.lb_child_organisms_capacity->setText(QString::fromStdString("Child organisms capacity: " + std::to_string(edc.stc.child_organisms.capacity())));
    ui.lb_child_organisms_in_use  ->setText(QString::fromStdString("Child organisms in use: " + std::to_string(edc.stc.child_organisms.size() - edc.stc.free_child_organisms_positions.size())));
    ui.lb_dead_organisms          ->setText(QString::fromStdString("Dead organisms: " + std::to_string(edc.stc.dead_organisms_positions.size())));
    ui.lb_organisms_capacity      ->setText(QString::fromStdString("Organisms capacity: " + std::to_string(edc.stc.organisms.capacity())));
    ui.lb_total_organisms         ->setText(QString::fromStdString("Total organisms: " + std::to_string(edc.stc.organisms.size())));
    ui.lb_last_alive_position     ->setText(QString::fromStdString("Last alive position: " + std::to_string(edc.stc.last_alive_position)));
    ui.lb_dead_inside             ->setText(QString::fromStdString("Dead inside: " + std::to_string(edc.stc.dead_organisms_before_last_alive_position)));
    ui.lb_dead_outside            ->setText(QString::fromStdString("Dead outside: " + std::to_string(edc.stc.num_dead_organisms - edc.stc.dead_organisms_before_last_alive_position)));

    ui.lb_zoom       ->setText(QString::fromStdString("Zoom: " + std::to_string(scaling_zoom)));
    ui.lb_viewpoint_x->setText(QString::fromStdString("Viewpoint x: " + std::to_string(center_x)));
    ui.lb_viewpoint_y->setText(QString::fromStdString("Viewpoint y: " + std::to_string(center_y)));

    for (int n = 0; n < 3; n++) {
        for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
            labels[n][i]->setText(QString::fromStdString("Avg " + ORGANISM_BLOCK_NAMES[i] + " num: " + to_str(info.avgs[n].block_avgs[i])));
        }
    }
}