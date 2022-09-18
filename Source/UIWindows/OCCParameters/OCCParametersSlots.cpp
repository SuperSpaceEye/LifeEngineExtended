//
// Created by spaceeye on 18.09.22.
//

#include "OCCParameters.h"


void OCCParametersWindow::cb_use_uniform_group_size_slot      (bool state) { occp.uniform_group_size_distribution = state;}

void OCCParametersWindow::cb_use_uniform_move_distance_slot   (bool state) { occp.uniform_move_distance = state;}

void OCCParametersWindow::cb_use_uniform_mutation_type_slot   (bool state) { occp.uniform_mutation_distribution = state;}

void OCCParametersWindow::cb_use_uniform_occ_instructions_slot(bool state) { occp.uniform_occ_instructions_mutation = state;}



void OCCParametersWindow::le_max_group_size_slot() {
    engine.pause();

    le_slot_lower_bound(occp.max_group_size, occp.max_group_size, "int", ui.le_max_group_size, 1, "1");
    occp.group_size_weights.resize(occp.max_group_size, 1);

    engine.unpause();

    create_group_size_distribution();
}

void OCCParametersWindow::le_max_move_distance_slot() {
    engine.pause();

    le_slot_lower_bound(occp.max_distance, occp.max_distance, "int", ui.le_max_move_distance, 1, "1");
    occp.move_distance_mutation_weights.resize(occp.max_distance, 1);

    engine.unpause();

    create_move_distance_distribution();
}