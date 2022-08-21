//
// Created by spaceeye on 21.08.22.
//

#include "OrganismsController.h"

Organism *OrganismsController::get_child_organism(EngineDataContainer &edc) {
    if (!edc.stc.free_child_organisms_positions.empty()) {
        auto * ptr = &edc.stc.child_organisms[edc.stc.free_child_organisms_positions.back()];
        edc.stc.free_child_organisms_positions.pop_back();
        return ptr;
    }

    edc.stc.child_organisms.emplace_back(Organism());
    edc.stc.child_organisms.back().array_place = edc.stc.child_organisms.size()-1;
    return & edc.stc.child_organisms.back();
}

void OrganismsController::free_child_organism(Organism *child_organism, EngineDataContainer &edc) {
    edc.stc.free_child_organisms_positions.emplace_back(child_organism->array_place);
}

void OrganismsController::emplace_child_organisms_to_main_vector(Organism *child_organism, EngineDataContainer &edc) {
    auto * main_organism_ptr = get_main_organism(edc);

    auto main_organism_place = main_organism_ptr->array_place;
    *main_organism_ptr = *child_organism;
    main_organism_ptr->array_place = main_organism_place;

    free_child_organism(child_organism, edc);
    main_organism_ptr->is_dead = false;
}

Organism *OrganismsController::get_main_organism(EngineDataContainer &edc) {
    // If there are organisms not in use, take them.
    if (!edc.stc.dead_organisms_positions.empty()) {
        auto * ptr = &edc.stc.organisms[edc.stc.dead_organisms_positions.back()];
        edc.stc.dead_organisms_positions.pop_back();
        return ptr;
    }

    // If there are no free organisms, create default one and return it.
    edc.stc.organisms.emplace_back(Organism());
    edc.stc.organisms.back().array_place = edc.stc.organisms.size()-1;
    return & edc.stc.organisms.back();
}

void OrganismsController::precise_sort_dead_organisms(EngineDataContainer &edc) {
    //TODO the result of sorting doesn't need to be perfect, just good enough.
    std::sort(edc.stc.dead_organisms_positions.begin(), edc.stc.dead_organisms_positions.end(), [](uint32_t a, uint32_t b) {
        return a < b;
    });
}

void OrganismsController::check_dead_to_alive_organisms_factor(EngineDataContainer &edc) {
    if ((edc.stc.organisms.size() - edc.stc.num_dead_organisms) * edc.stc.max_dead_to_alive_organisms_factor < edc.stc.organisms.size()) {
        return;
    }

    int dead_organisms = 0;
    int last_alive_organism_place = edc.stc.organisms.size()-1;
    for (; last_alive_organism_place >= 0; last_alive_organism_place--) {
        auto & organism = edc.stc.organisms[last_alive_organism_place];
        if (!organism.is_dead) {break;}
        dead_organisms++;
    }

    edc.stc.organisms.erase(edc.stc.organisms.begin() + last_alive_organism_place + 1, edc.stc.organisms.end());
    edc.stc.dead_organisms_positions.erase(edc.stc.dead_organisms_positions.end() + 1 - dead_organisms, edc.stc.dead_organisms_positions.end());
    edc.stc.organisms.shrink_to_fit();
    edc.stc.dead_organisms_positions.shrink_to_fit();
}