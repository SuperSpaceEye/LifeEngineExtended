//
// Created by spaceeye on 21.08.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H
#define LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H

#include <algorithm>

#include "../Containers/CPU/EngineDataContainer.h"
#include "../Organism/CPU/Organism.h"

class OrganismsController {
    static Organism *get_child_organism(EngineDataContainer &edc);
    static void free_child_organism(Organism *child_organism, EngineDataContainer &edc);
    static void emplace_child_organisms_to_main_vector(Organism * child_organism, EngineDataContainer &edc);
    static Organism *get_main_organism(EngineDataContainer &edc);
    static void precise_sort_dead_organisms(EngineDataContainer &edc);
    static void check_dead_to_alive_organisms_factor(EngineDataContainer &edc);
};


#endif //LIFEENGINEEXTENDED_ORGANISMSCONTROLLER_H
