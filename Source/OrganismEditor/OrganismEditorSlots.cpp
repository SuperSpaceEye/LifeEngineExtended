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

    reset_scale_view();
    clear_grid();
    create_image();
}

void OrganismEditor::b_load_organism_slot() {
    QString selected_filter;
    auto file_name = QFileDialog::getOpenFileName(this, tr("Load organism"), "",
                                                  tr("Custom save type (*.lfeo);;JSON (*.json)"), &selected_filter);
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.lfeo)") {
        filetype = ".lfeo";
    } else if (selected_filter.toStdString() == "JSON (*.json)"){
        filetype = ".json";
    } else {
        return;
    }

    std::string full_path = file_name.toStdString();

    try {
        if (filetype == ".lfeo") {
            std::ifstream in(full_path, std::ios::in | std::ios::binary);
            read_organism(in);
            in.close();

        } else if (filetype == ".json") {
            read_json_organism(full_path);
        }
    } catch (std::string & _) {
        display_message("Loading of organism was unsuccessful.");
    }

    update_chosen_organism();
}

void OrganismEditor::b_save_organism_slot() {
    QString selected_filter;
    QFileDialog file_dialog{};

    auto file_name = file_dialog.getSaveFileName(this, tr("Save organism"), "",
                                                 "Custom save type (*.lfeo);;JSON (*.json)", &selected_filter);
#ifndef __WIN32
    bool file_exists = std::filesystem::exists(file_name.toStdString());
#endif
    std::string filetype;
    if (selected_filter.toStdString() == "Custom save type (*.lfeo)") {
        filetype = ".lfeo";
    } else if (selected_filter.toStdString() == "JSON (*.json)") {
        filetype = ".json";
    } else {
        return;
    }
    std::string full_path = file_name.toStdString();

#ifndef __WIN32
    if (!file_exists) {
        full_path = file_name.toStdString() + filetype;
    }
#endif

    if (filetype == ".lfeo") {
        std::ofstream out(full_path, std::ios::out | std::ios::binary);
        write_organism(out);
        out.close();

    } else {
        write_json_organism(full_path);
    }
}

void OrganismEditor::b_reset_organism_slot() {
    for (int i = 0; i < editor_organism->anatomy->_organism_blocks.size(); i++) {
        if (editor_organism->anatomy->_organism_blocks[i].relative_x == 0 && editor_organism->anatomy->_organism_blocks[i].relative_y == 0) {
            continue;
        }
        editor_organism->anatomy->_organism_blocks.erase(editor_organism->anatomy->_organism_blocks.begin() + i);
        i--;
    }

    editor_organism->anatomy->set_many_blocks(editor_organism->anatomy->_organism_blocks);
    update_chosen_organism();
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
    finalize_chosen_organism();
    _parent_ui->rb_null_button->setChecked(true);
}

void OrganismEditor::rb_choose_organism_slot() {
    display_message("This function is buggy right now and can crash program.");
    *c_mode = CursorMode::ChooseOrganism;
    _parent_ui->rb_null_button->setChecked(true);
}

void OrganismEditor::rb_edit_anatomy_slot() {
    _ui.stackedWidget->setCurrentIndex(0);
}

void OrganismEditor::rb_edit_brain_slot() {
    _ui.stackedWidget->setCurrentIndex(1);
}

//==================== Combo boxes ====================


//TODO i don't fkn know why left and right is switched
void OrganismEditor::cmd_block_rotation_slot(QString name) {
    if (name.toStdString() == "Up")    {chosen_block_rotation = Rotation::UP;}
    if (name.toStdString() == "Left")  {chosen_block_rotation = Rotation::RIGHT;}
    if (name.toStdString() == "Down")  {chosen_block_rotation = Rotation::DOWN;}
    if (name.toStdString() == "Right") {chosen_block_rotation = Rotation::LEFT;}
}

void OrganismEditor::cmd_organism_rotation_slot(QString name) {
    if (name.toStdString() == "Up")    {editor_organism->rotation = Rotation::UP;}
    if (name.toStdString() == "Left")  {editor_organism->rotation = Rotation::RIGHT;}
    if (name.toStdString() == "Down")  {editor_organism->rotation = Rotation::DOWN;}
    if (name.toStdString() == "Right") {editor_organism->rotation = Rotation::LEFT;}
    finalize_chosen_organism();
}