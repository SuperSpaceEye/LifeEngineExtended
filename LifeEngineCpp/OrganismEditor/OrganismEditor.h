//
// Created by spaceeye on 25.06.22.
//

#ifndef THELIFEENGINECPP_ORGANISMEDITOR_H
#define THELIFEENGINECPP_ORGANISMEDITOR_H

#include <vector>

#include "../Linspace.h"
#include "../Organism/CPU/Organism.h"
#include "../Organism/CPU/Anatomy.h"
#include "../Organism/CPU/Brain.h"
#include "../Organism/CPU/Rotation.h"
#include "../GridBlocks/BaseGridBlock.h"
#include "EditorUI.h"
#include "../MainWindow/WindowUI.h"

struct EditBlock : BaseGridBlock {
    //For when cursor is hovering above block
    bool hovering = false;
    explicit EditBlock(BlockTypes type, Rotation rotation = Rotation::UP) :
            BaseGridBlock(type, rotation){}
};

class OrganismEditor: public QWidget {
    Q_OBJECT
private:
    Ui::Editor _ui;
    Ui::MainWindow * _parent_ui = nullptr;

    std::vector<std::vector<EditBlock>> edit_grid;
    Organism organism;

    float scaling_coefficient = 1.2;
    float scaling_zoom = 1;

    float center_x;
    float center_y;

    int editor_width;
    int editor_height;

    int image_width = 0;
    int image_height = 0;

    void closeEvent(QCloseEvent * event) override;
public:
    OrganismEditor()=default;

    void init(int width, int height, Ui::MainWindow *parent_ui);

    void set_block(int x, int y, BaseGridBlock block);
    BaseGridBlock get_block(int x, int y);
    void set_organism(Organism organism);
    Organism get_organism();

    void resize_editing_grid(int width, int height);
    void resize_image_editing_grid(int width, int height);
    std::vector<unsigned char> create_image();

    void move_center(int delta_x, int delta_y){};
    void reset_scale_view();
};

#endif //THELIFEENGINECPP_ORGANISMEDITOR_H