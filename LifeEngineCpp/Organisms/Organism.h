//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ORGANISM_H
#define THELIFEENGINECPP_ORGANISM_H

#include "Anatomy.h"

class Organism {
private:
    //coordinates of a central block of a cell
    int x;
    int y;
    int lifetime = 0;
    int food_collected = 0;
    //TODO
    bool can_rotate = false;

    Anatomy anatomy;

public:
    void inherit(Organism* parent);

};


#endif //THELIFEENGINECPP_ORGANISM_H
