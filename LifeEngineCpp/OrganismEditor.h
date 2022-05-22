//
// Created by spaceeye on 21.05.2022.
//

#ifndef THELIFEENGINECPP_ORGANISMEDITOR_H
#define THELIFEENGINECPP_ORGANISMEDITOR_H

#include <vector>

#include "Linspace.h"
#include "Organism/Organism.h"
#include "Organism/Anatomy.h"
#include "Organism/Brain.h"
#include "Organism/Rotation.h"

struct EditBlock : BaseGridBlock {
    //For when cursor is hovering above block
    bool hovering = false;
    explicit EditBlock(BlockTypes type, Rotation rotation = Rotation::UP, Neighbors neighbors = Neighbors{}) :
    BaseGridBlock(type, rotation, neighbors){}
};

class OrganismEditor {
    std::vector<std::vector<EditBlock>> edit_grid;
    Organism organism;

    float scaling_coefficient = 1.2;
    float scaling_zoom = 1;

    float center_x;
    float center_y;

    int image_width = 0;
    int image_height = 0;
public:
    OrganismEditor(int width, int height, int image_width, int image_height);
    OrganismEditor()=default;

    void set_block(int x, int y, OrganismBlock block);
    OrganismBlock get_block(int x, int y);
    void set_organism(Organism organism);
    Organism get_organism();

    void resize_editing_grid(int width, int height);
    void resize_image_editing_grid(int width, int height);
    std::vector<unsigned char> create_image();

    void move_center(int delta_x, int delta_y){};
    void reset_scale_view();
};


#endif //THELIFEENGINECPP_ORGANISMEDITOR_H
