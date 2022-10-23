// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 25.06.22.
//

#ifndef THELIFEENGINECPP_ORGANISMEDITOR_H
#define THELIFEENGINECPP_ORGANISMEDITOR_H

#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>

#include <QGraphicsPixmapItem>
#include <QTimer>
#include <QWheelEvent>
#include <QFileDialog>

#include "../../Stuff/rapidjson/document.h"
#include "../../Stuff/rapidjson/writer.h"
#include "../../Stuff/rapidjson/stringbuffer.h"

#include "../../Stuff/Linspace.h"
#include "../../Organism/CPU/Organism.h"
#include "../../Organism/CPU/Anatomy.h"
#include "../../Organism/CPU/Brain.h"
#include "../../Organism/CPU/Rotation.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "EditorUI.h"
#include "../MainWindow/WindowUI.h"
#include "../../Stuff/Vector2.h"
#include "../../Containers/CPU/ColorContainer.h"
#include "../../Stuff/textures.h"
#include "../../Stuff/CursorMode.h"
#include "../../Stuff/MiscFuncs.h"
#include "../../Stuff/ImageCreation.h"
#include "../../Stuff/DataSavingFunctions.h"
#include "OCCTranspiler/OCCTranspiler.h"
#include "../../Stuff/ImageCreation.h"

#ifdef __CUDA_USED__
#include "../../Stuff/cuda_image_creator.cuh"
#endif

class OrganismEditor: public QWidget {
    Q_OBJECT
public:
    std::vector<std::string> decisions{"Do Nothing", "Go Away", "Go Towards"};
    std::vector<std::string> observations{"Mouth Cell", "Producer Cell", "Mover Cell", "Killer Cell", "Armor Cell", "Eye Cell", "Food", "Wall"};
    std::map<std::string, std::map<std::string, QCheckBox*>> brain_checkboxes;
    std::map<std::string, QLineEdit*> brain_line_edits;
    std::map<std::string, BlockTypes>     mapped_block_types_s_to_type;
    std::map<std::string, SimpleDecision> mapped_decisions_s_to_type;
    std::map<BlockTypes,     std::string> mapped_block_types_type_to_s;
    std::map<SimpleDecision, std::string> mapped_decisions_type_to_s;

    Ui::MainWindow * parent_ui = nullptr;

    std::vector<BaseGridBlock> edit_grid;
    std::vector<unsigned char> edit_image;
    CursorMode * c_mode = nullptr;

    TexturesContainer & textures;

    SimulationParameters * sp;
    OrganismBlockParameters * bp;

    ColorContainer * color_container;
    OCCLogicContainer occl;
    OCCTranspiler occt{};

    double scaling_coefficient = 1.2;

    double center_x;
    double center_y;

    int new_editor_width = 15;
    int new_editor_height = 15;

    QGraphicsPixmapItem pixmap_item;
    QGraphicsScene scene;

    std::chrono::time_point<std::chrono::milliseconds> point;

    void closeEvent(QCloseEvent * event) override;
    void resizeEvent(QResizeEvent * event) override;
    void wheelEvent(QWheelEvent *event) override;

    void place_organism_on_a_grid();
    void clear_grid();

    void update_gui();
    void update_cell_count_label();
    void initialize_gui();

    void occ_mode(bool state);

    void brain_cb_chooser(std::string observation, std::string action);
    void brain_weight_chooser(std::string observation, QLineEdit * le);

    void update_brain_state();
    void update_brain_checkboxes();
    void update_brain_line_edits();

    QLabel actual_cursor;

    Ui::Editor ui{};

    int editor_width = 15;
    int editor_height = 15;
    double scaling_zoom = 1;

    bool change_disabled = false;
    bool short_instructions = false;

    Organism * editor_organism;
    Organism ** chosen_organism;

    BlockTypes chosen_block_type = BlockTypes::MouthBlock;
    Rotation chosen_block_rotation = Rotation::UP;

    Rotation choosen_rotation = Rotation::UP;

    const bool & cuda_is_available;
    const bool & use_cuda;

#ifdef __CUDA_USED__
    CUDAImageCreator cuda_image_creator{};
#endif

    OrganismEditor(int width, int height, Ui::MainWindow *parent_ui, ColorContainer *color_container,
                   SimulationParameters *sp, OrganismBlockParameters *bp, CursorMode *cursor_mode,
                   Organism **chosen_organism, TexturesContainer &textures, OCCLogicContainer *occl,
                   OCCParameters *occp, const bool &cuda_is_available, const bool &use_cuda);

    Vector2<int> calculate_cursor_pos_on_grid(int x, int y);

    void resize_editing_grid(int width, int height);
    void resize_image();

    void move_center(int delta_x, int delta_y);
    void reset_scale_view();

    void create_image();

    void finalize_chosen_organism();
    void load_chosen_organism();

    void read_organism(std::ifstream & is);

    void read_json_organism(std::string & full_path);
    void write_json_organism(std::string &full_path);

    void clear_occ();
    void load_occ();

    bool check_edit_area();

    void update_brain_edit_visibility(bool weighted_edits_visible);

private slots:
    void b_load_organism_slot();
    void b_reset_editing_view_slot();
    void b_resize_editing_grid_slot();
    void b_save_organism_slot();
    void b_reset_organism_slot();
    void b_compile_occ_slot();

    void le_anatomy_mutation_rate_slot();
    void le_grid_height_slot();
    void le_grid_width_slot();
    void le_move_range_slot();
    void le_brain_mutation_rate_slot();

    void rb_armor_slot();
    void rb_eye_slot();
    void rb_killer_slot();
    void rb_mouth_slot();
    void rb_mover_slot();
    void rb_producer_slot();
    void rb_edit_anatomy_slot();
    void rb_edit_brain_slot();
    void rb_edit_occ_slot();

    void cmd_block_rotation_slot(const QString& name);
    void cmd_organism_rotation_slot(const QString& name);

    void cb_short_instructions_slot(bool state);
public slots:
    void rb_place_organism_slot();
    void rb_choose_organism_slot();
};

#endif //THELIFEENGINECPP_ORGANISMEDITOR_H
