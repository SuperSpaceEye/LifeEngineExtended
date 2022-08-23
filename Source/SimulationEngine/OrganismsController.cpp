//
// Created by spaceeye on 21.08.22.
//

#include "OrganismsController.h"

//TODO this will not work because when resize of vector happens during push_back or emplace_back, all pointers to vector will become invalid.
Organism *OrganismsController::get_new_child_organism(EngineDataContainer &edc) {
    if (!edc.stc.free_child_organisms_positions.empty()) {
        auto * ptr = &edc.stc.child_organisms[edc.stc.free_child_organisms_positions.back()];
        edc.stc.free_child_organisms_positions.pop_back();
        return ptr;
    }

    edc.stc.child_organisms.emplace_back();
    edc.stc.child_organisms.back().vector_index = edc.stc.child_organisms.size() - 1;
    return & edc.stc.child_organisms.back();
}

void OrganismsController::free_child_organism(Organism *child_organism, EngineDataContainer &edc) {
    if (child_organism == nullptr) { return;}
    edc.stc.free_child_organisms_positions.emplace_back(child_organism->vector_index);
}

void OrganismsController::free_main_organism(Organism *organism, EngineDataContainer &edc) {
    free_child_organism(get_child_organism_by_index(organism->child_pattern_index, edc), edc);
    organism->child_pattern_index = -1;

    organism->is_dead = true;
    edc.stc.dead_organisms_positions.emplace_back(organism->vector_index);
    if (organism->vector_index == edc.stc.last_alive_position) {
        edc.stc.last_alive_position--;
    }

    edc.stc.num_dead_organisms++;
    edc.stc.num_alive_organisms--;
}

//returns index of placed organism
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

Organism *OrganismsController::get_new_main_organism(EngineDataContainer &edc) {
    // If there are organisms not in use, take them.
    edc.stc.num_alive_organisms++;
    if (!edc.stc.dead_organisms_positions.empty()) {
        edc.stc.num_dead_organisms--;
        auto * ptr = &edc.stc.organisms[edc.stc.dead_organisms_positions.back()];
        edc.stc.dead_organisms_positions.pop_back();
        return ptr;
    }

    // If there are no free organisms, create default one and return it.
    edc.stc.organisms.emplace_back();
    edc.stc.organisms.back().vector_index = edc.stc.organisms.size() - 1;
    edc.stc.last_alive_position = edc.stc.organisms.size()-1;
    return & edc.stc.organisms.back();
}

void OrganismsController::precise_sort_dead_organisms(EngineDataContainer &edc) {
    //TODO the result of sorting doesn't need to be perfect, just good enough.
    std::sort(edc.stc.dead_organisms_positions.begin(), edc.stc.dead_organisms_positions.end(), [](uint32_t a, uint32_t b) {
        return a > b;
    });
}

//TODO i probably messed something up here.
void OrganismsController::check_dead_to_alive_organisms_factor(EngineDataContainer &edc) {
    if (edc.stc.num_dead_organisms <= edc.stc.num_alive_organisms * edc.stc.max_dead_to_alive_organisms_factor || edc.stc.num_alive_organisms == 0) {
        return;
    }
    std::sort(edc.stc.dead_organisms_positions.begin(), edc.stc.dead_organisms_positions.end(), [](uint32_t a, uint32_t b) {
        return a < b;
    });

    uint32_t last_alive_organism_place = get_last_alive_organism_position(edc);
    uint32_t dead_organisms = edc.stc.organisms.size() - 1 - last_alive_organism_place;

    edc.stc.num_dead_organisms -= dead_organisms;

    edc.stc.organisms.erase(edc.stc.organisms.begin() + last_alive_organism_place + 1, edc.stc.organisms.end());
    edc.stc.dead_organisms_positions.erase(edc.stc.dead_organisms_positions.end() - dead_organisms, edc.stc.dead_organisms_positions.end());

    edc.stc.organisms.shrink_to_fit();
    edc.stc.dead_organisms_positions.shrink_to_fit();
}

int32_t OrganismsController::get_last_alive_organism_position(EngineDataContainer &edc) {
    int32_t last_alive_organism_place = edc.stc.organisms.size() - 1;
    while (edc.stc.organisms[last_alive_organism_place].is_dead) {
        last_alive_organism_place--;
    }

    return last_alive_organism_place;
}