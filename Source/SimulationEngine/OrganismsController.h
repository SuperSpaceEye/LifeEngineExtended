//
// Created by spaceeye on 21.08.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H
#define LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H

#include <algorithm>

#include "../Containers/CPU/EngineDataContainer.h"
#include "../Organism/CPU/Organism.h"

class OrganismsController {
public:
    //Either returns free organism or allocates new one
    static Organism *get_new_child_organism(EngineDataContainer &edc);
    static void free_child_organism(Organism *child_organism, EngineDataContainer &edc);
    //Moves data of child organism to new main organism and returns position of new organism
    static int32_t emplace_child_organisms_to_main_vector(Organism * child_organism, EngineDataContainer &edc);
    //Either returns dead organism or allocates new one
    static Organism *get_new_main_organism(EngineDataContainer &edc);
    static void free_main_organism(Organism * organism, EngineDataContainer &edc);
    //Sorts locations of dead organisms in main vector from smallest to biggest
    static void precise_sort_low_to_high_dead_organisms_positions(EngineDataContainer &edc);
    static void precise_sort_high_to_low_dead_organisms_positions(EngineDataContainer &edc);
    //Dead organisms can be between alive ones.
    static int32_t get_last_alive_organism_position(EngineDataContainer &edc);
    static inline Organism *get_organism_by_index(int32_t organism_index, EngineDataContainer &edc)             {return organism_index < 0 ? nullptr : &edc.stc.organisms[organism_index];}
    static inline Organism *get_child_organism_by_index(int32_t child_organism_index, EngineDataContainer &edc) {return child_organism_index < 0 ? nullptr : &edc.stc.child_organisms[child_organism_index];}
    static void compress_organisms(EngineDataContainer &edc);
};


#endif //LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H
