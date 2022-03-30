//
// Created by spaceeye on 23.03.2022.
//

#ifndef THELIFEENGINECPP_SPECIES_H
#define THELIFEENGINECPP_SPECIES_H

#include "Anatomy.h"
#include "Organism.h"
#include "iostream"
#include "vector"

//TODO think about that
class Species {
private:
    Anatomy species_anatomy;
    std::vector<Organism*> species_organisms;

    //will check if organisms of this species are all dead.
    //if species is extinct, then log statistic (maybe) and call destructor.
    void check_if_extinct();

};


#endif //THELIFEENGINECPP_SPECIES_H
