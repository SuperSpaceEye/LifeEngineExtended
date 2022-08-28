//
// Created by spaceeye on 21.08.22.
//

#include "OrganismsController.h"

Organism *OrganismsController::get_new_child_organism(EngineDataContainer &edc) {
    if (!edc.stc.free_child_organisms_positions.empty()) {
        auto * ptr = &edc.stc.child_organisms[edc.stc.free_child_organisms_positions.back()];
        edc.stc.free_child_organisms_positions.pop_back();
        return ptr;
    }

    if (edc.stc.child_organisms.size() == edc.stc.child_organisms.capacity()) {
        edc.stc.child_organisms.reserve(edc.stc.child_organisms.capacity()*edc.stc.memory_allocation_strategy_modifier);
        edc.stc.free_child_organisms_positions.reserve(edc.stc.free_child_organisms_positions.capacity()*edc.stc.memory_allocation_strategy_modifier);
    }

    edc.stc.child_organisms.emplace_back();

    edc.stc.child_organisms.back().vector_index = edc.stc.child_organisms.size() - 1;
    return & edc.stc.child_organisms.back();
}

void OrganismsController::free_child_organism(Organism *child_organism, EngineDataContainer &edc) {
    if (child_organism == nullptr) { return;}
    edc.stc.free_child_organisms_positions.emplace_back(child_organism->vector_index);
}



Organism *OrganismsController::get_new_main_organism(EngineDataContainer &edc) {
    // If there are organisms not in use, take them.
    edc.stc.num_alive_organisms++;
    if (!edc.stc.dead_organisms_positions.empty()) {
        edc.stc.num_dead_organisms--;
        auto * ptr = &edc.stc.organisms[edc.stc.dead_organisms_positions.back()];
        edc.stc.dead_organisms_positions.pop_back();

        if (ptr->vector_index < edc.stc.last_alive_position) {
            edc.stc.dead_organisms_before_last_alive_position--;
        }

        return ptr;
    }

    if (edc.stc.organisms.size() == edc.stc.organisms.capacity()) {
        edc.stc.organisms.reserve(edc.stc.organisms.capacity()*edc.stc.memory_allocation_strategy_modifier);
        edc.stc.dead_organisms_positions.reserve(edc.stc.dead_organisms_positions.capacity()*edc.stc.memory_allocation_strategy_modifier);
    }

    // If there are no free organisms, create default one and return it.
    edc.stc.organisms.emplace_back();
    edc.stc.organisms.back().vector_index = edc.stc.organisms.size() - 1;
    edc.stc.last_alive_position = edc.stc.organisms.size()-1;
    return & edc.stc.organisms.back();
}

void OrganismsController::free_main_organism(Organism *organism, EngineDataContainer &edc) {
    free_child_organism(get_child_organism_by_index(organism->child_pattern_index, edc), edc);
    organism->child_pattern_index = -1;

    organism->is_dead = true;
    edc.stc.dead_organisms_positions.emplace_back(organism->vector_index);

    if (organism->vector_index < edc.stc.last_alive_position) {
        edc.stc.dead_organisms_before_last_alive_position++;
    }

    edc.stc.num_dead_organisms++;
    edc.stc.num_alive_organisms--;
}

int32_t OrganismsController::emplace_child_organisms_to_main_vector(Organism *child_organism, EngineDataContainer &edc) {
    auto * main_o_ptr = get_new_main_organism(edc);

    auto main_organism_place = main_o_ptr->vector_index;
    main_o_ptr->move_organism(*child_organism);
    main_o_ptr->vector_index = main_organism_place;

    if (main_o_ptr->vector_index > edc.stc.last_alive_position) { edc.stc.last_alive_position = main_o_ptr->vector_index; }

    free_child_organism(child_organism, edc);
    main_o_ptr->is_dead = false;
    main_o_ptr->init_values();

    return main_o_ptr->vector_index;
}

void OrganismsController::precise_sort_low_to_high_dead_organisms_positions(EngineDataContainer &edc) {
    std::sort(edc.stc.dead_organisms_positions.begin(), edc.stc.dead_organisms_positions.end(), [](uint32_t a, uint32_t b) {
        return a < b;
    });
}

void OrganismsController::precise_sort_high_to_low_dead_organisms_positions(EngineDataContainer &edc) {
    std::sort(edc.stc.dead_organisms_positions.begin(), edc.stc.dead_organisms_positions.end(), [](uint32_t a, uint32_t b) {
        return a > b;
    });
}

int32_t OrganismsController::get_last_alive_organism_position(EngineDataContainer &edc) {
    int32_t last_alive_organism_place = edc.stc.organisms.size() - 1;
    while (edc.stc.organisms[last_alive_organism_place].is_dead && last_alive_organism_place > 0) {
        last_alive_organism_place--;
    }

    return last_alive_organism_place;
}

//Will compress alive organisms so that there will be no dead organisms between alive ones.
//Dead organisms positions should be sorted first.
void OrganismsController::compress_organisms(EngineDataContainer &edc) {
    if (edc.stc.dead_organisms_before_last_alive_position * edc.stc.max_dead_organisms_in_alive_section_factor < edc.stc.num_alive_organisms) { return;}
    if (edc.stc.num_alive_organisms == 1) { return;}

    //TODO on call of compress_organisms go through all dead organisms positions and only append to temp list positions that are < last_alive_position
    OrganismsController::precise_sort_high_to_low_dead_organisms_positions(edc);

    int left_index = 0;
    int right_index = edc.stc.last_alive_position;

    Organism * left_organism;
    Organism * right_organism;

    //there will be no dead organisms before last_alive_position by the end of compression
    edc.stc.dead_organisms_before_last_alive_position = 0;

    while (left_index <= right_index) {
        //while right organism is dead, go left until it is alive
        while ((right_organism = &edc.stc.organisms[right_index])->is_dead) {right_index--; if (right_index <= left_index) { goto endlogic_right;}}
        //while left organism is alive, go right until it is dead
        while (!(left_organism = &edc.stc.organisms[left_index])->is_dead)  {left_index++;  if (left_index >= right_index) { goto endlogic_left;}}

        //If left organism is dead, and right one is alive, then swap them, update positions, and repeat the process.
        std::swap(*right_organism, *left_organism);
        right_organism->vector_index = right_index;
        left_organism->vector_index  = left_index;

        edc.stc.dead_organisms_positions.pop_back();
        edc.stc.temp_dead_organisms_positions.emplace_back(right_index);
    }
    endlogic_left:
    //left index by the end of compression will be the last alive position
    edc.stc.last_alive_position = left_index;
    endlogic_right:
    edc.stc.dead_organisms_positions.insert(edc.stc.dead_organisms_positions.end(),
                                            edc.stc.temp_dead_organisms_positions.begin(),
                                            edc.stc.temp_dead_organisms_positions.end());

    edc.stc.temp_dead_organisms_positions.clear();
}