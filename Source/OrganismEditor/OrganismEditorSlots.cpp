// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 26.06.22.
//

#include "OrganismEditor.h"

//==================== Buttons ====================

void OrganismEditor::b_reset_editing_view_slot() {
    reset_scale_view();
    create_image();
}

void OrganismEditor::b_resize_editing_grid_slot() {
    resize_editing_grid(new_editor_width, new_editor_height);
//
//    editor_organism->x = new_editor_width  / 2;
//    editor_organism->y = new_editor_height / 2;

//    auto anatomy = std::make_shared<Anatomy>();
//    anatomy->set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
//
//    auto brain = std::make_shared<Brain>(editor_organism->brain);
//
//    auto * _editor_organism = new Organism(editor_width  / 2,
//                                           editor_height / 2,
//                                           Rotation::UP,
//                                           anatomy,
//                                           brain,
//                                           editor_organism->sp,
//                                           editor_organism->bp,
//                                           editor_organism->move_range);
//
//    _editor_organism->anatomy_mutation_rate = editor_organism->anatomy_mutation_rate;
//    _editor_organism->brain_mutation_rate = editor_organism->brain_mutation_rate;
//    _editor_organism->max_do_nothing_lifetime = editor_organism->max_do_nothing_lifetime;
//    _editor_organism->max_decision_lifetime = editor_organism->max_decision_lifetime;
//    _editor_organism->rotation = editor_organism->rotation;
//
//    std::swap(editor_organism, _editor_organism);
//
//    delete _editor_organism;

    reset_scale_view();
    clear_grid();
    create_image();
}

void OrganismEditor::b_load_organism_slot() {

}

void OrganismEditor::b_save_organism_slot() {

}

//==================== Line edits ====================


void OrganismEditor::le_anatomy_mutation_rate_slot() {
    float fallback = editor_organism->anatomy_mutation_rate;
    auto result = try_convert_message_box_template<float>("Inputted text is not a float", _ui.le_anatomy_mutation_rate,
                                                          fallback);
    if (!result.is_valid) {return;}
    if (result.result < 0) {display_message("Input cannot be less than 0."); return;}
    if (result.result > 1) {display_message("Input cannot be more than 1."); return;}
    editor_organism->anatomy_mutation_rate = result.result;
}

void OrganismEditor::le_grid_width_slot() {
    int fallback = new_editor_width;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_grid_width, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 5) {display_message("Input cannot be less than 10."); return;}
    new_editor_width = result.result;
}

void OrganismEditor::le_grid_height_slot() {
    int fallback = new_editor_height;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int.", _ui.le_grid_height, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 5) {display_message("Input cannot be less than 10."); return;}
    new_editor_height = result.result;
}

void OrganismEditor::le_move_range_slot() {
    int fallback = editor_organism->move_range;
    auto result = try_convert_message_box_template<int>("Inputted text is not an int", _ui.le_move_range, fallback);
    if (!result.is_valid) {return;}
    if (result.result < 1) {display_message("Input cannot be less than 1."); return;}
    editor_organism->move_range = result.result;
}

//==================== Radio buttons ====================


void OrganismEditor::rb_armor_slot    () { chosen_block_type = BlockTypes::ArmorBlock    ;}

void OrganismEditor::rb_eye_slot      () { chosen_block_type = BlockTypes::EyeBlock      ;}

void OrganismEditor::rb_killer_slot   () { chosen_block_type = BlockTypes::KillerBlock   ;}

void OrganismEditor::rb_mouth_slot    () { chosen_block_type = BlockTypes::MouthBlock    ;}

void OrganismEditor::rb_mover_slot    () { chosen_block_type = BlockTypes::MoverBlock    ;}

void OrganismEditor::rb_producer_slot () { chosen_block_type = BlockTypes::ProducerBlock ;}

void OrganismEditor::rb_place_organism_slot() {
    *c_mode = CursorMode::PlaceOrganism;
    _parent_ui->rb_null_button->setChecked(true);
}

void OrganismEditor::rb_choose_organism_slot() {
    *c_mode = CursorMode::ChooseOrganism;
    _parent_ui->rb_null_button->setChecked(true);
}

void OrganismEditor::rb_edit_anatomy_slot() {
    _ui.stackedWidget->setCurrentIndex(0);
}

void OrganismEditor::rb_edit_brain_slot() {
    _ui.stackedWidget->setCurrentIndex(1);
}