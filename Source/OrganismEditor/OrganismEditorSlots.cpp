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
    auto file_name = QFileDialog::getOpenFileName(this, tr("Load organism_index"), "",
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
        display_message("Loading of organism_index was unsuccessful.");
    }
    finalize_chosen_organism();
}

void OrganismEditor::b_save_organism_slot() {
    QString selected_filter;
    QFileDialog file_dialog{};

    auto file_name = file_dialog.getSaveFileName(this, tr("Save organism_index"), "",
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
        DataSavingFunctions::write_organism(out, editor_organism);
        out.close();

    } else {
        rapidjson::Document j_organism;
        j_organism.SetObject();

        write_json_organism(full_path);
    }
}

void OrganismEditor::b_reset_organism_slot() {
    for (int i = 0; i < editor_organism->anatomy._organism_blocks.size(); i++) {
        if (editor_organism->anatomy._organism_blocks[i].relative_x == 0 && editor_organism->anatomy._organism_blocks[i].relative_y == 0) {
            continue;
        }
        editor_organism->anatomy._organism_blocks.erase(editor_organism->anatomy._organism_blocks.begin() + i);
        i--;
    }

    editor_organism->anatomy.set_many_blocks(editor_organism->anatomy._organism_blocks);
    finalize_chosen_organism();
    create_image();
}

//==================== Line edits ====================


void OrganismEditor::le_anatomy_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(editor_organism->anatomy_mutation_rate, editor_organism->anatomy_mutation_rate, "float",
                                     ui.le_anatomy_mutation_rate, 0, "0", 1, "1");
}

void OrganismEditor::le_brain_mutation_rate_slot() {
    le_slot_lower_upper_bound<float>(editor_organism->brain_mutation_rate, editor_organism->brain_mutation_rate, "float",
                                     ui.le_brain_mutation_rate, 0, "0", 1, "1");
}

void OrganismEditor::le_grid_width_slot() {
    le_slot_lower_bound<int>(new_editor_width, new_editor_width, "int",
                             ui.le_grid_width, 5, "5");
}

void OrganismEditor::le_grid_height_slot() {
    le_slot_lower_bound<int>(new_editor_height, new_editor_height, "int",
                             ui.le_grid_height, 5, "5");
}

void OrganismEditor::le_move_range_slot() {
    le_slot_lower_bound<int>(editor_organism->move_range, editor_organism->move_range, "int",
                             ui.le_move_range, 1, "1");
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
    parent_ui->rb_null_button->setChecked(true);
    create_image();
}

void OrganismEditor::rb_choose_organism_slot() {
    *c_mode = CursorMode::ChooseOrganism;
    parent_ui->rb_null_button->setChecked(true);
}

void OrganismEditor::rb_edit_anatomy_slot() {
    ui.stackedWidget->setCurrentIndex(0);
}

void OrganismEditor::rb_edit_brain_slot() {
    ui.stackedWidget->setCurrentIndex(1);
}

//==================== Combo boxes ====================

void OrganismEditor::cmd_block_rotation_slot(const QString& name) {
    if (name.toStdString() == "Up")    {chosen_block_rotation = Rotation::UP;}
    if (name.toStdString() == "Left")  {chosen_block_rotation = Rotation::LEFT;}
    if (name.toStdString() == "Down")  {chosen_block_rotation = Rotation::DOWN;}
    if (name.toStdString() == "Right") {chosen_block_rotation = Rotation::RIGHT;}
}

void OrganismEditor::cmd_organism_rotation_slot(const QString& name) {
    if (name.toStdString() == "Up")    {editor_organism->rotation = Rotation::UP;}
    if (name.toStdString() == "Left")  {editor_organism->rotation = Rotation::RIGHT;}
    if (name.toStdString() == "Down")  {editor_organism->rotation = Rotation::DOWN;}
    if (name.toStdString() == "Right") {editor_organism->rotation = Rotation::LEFT;}
    finalize_chosen_organism();
}