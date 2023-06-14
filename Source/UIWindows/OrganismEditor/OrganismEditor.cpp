// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 25.06.22.
//

#include "OrganismEditor.h"

OrganismEditor::OrganismEditor(int width, int height, Ui::MainWindow *parent_ui, ColorContainer *color_container,
                               SimulationParameters *sp, OrganismBlockParameters *bp, CursorMode *cursor_mode,
                               Organism *chosen_organism, Textures::TexturesContainer &textures, OCCLogicContainer *occl,
                               OCCParameters *occp, const bool &cuda_is_available, const bool &use_cuda)
        : editor_width(width), editor_height(height),
                                parent_ui(parent_ui), color_container(color_container), sp(sp), bp(bp), c_mode(cursor_mode),
                                chosen_organism(chosen_organism), textures(textures), cuda_is_available(cuda_is_available),
                                use_cuda(use_cuda){
    ui.setupUi(this);

    ui.editor_graphicsView->show();
    ui.editor_graphicsView->setEnabled(false);

    scene.addItem(&pixmap_item);
    ui.editor_graphicsView->setScene(&scene);

    auto anatomy = Anatomy();
    anatomy.set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);
    anatomy.killer_mask.emplace_back(0);
    anatomy.eating_mask.emplace_back(1);

    auto brain = Brain();
    brain.brain_type = BrainTypes::SimpleBrain;

    auto occ = OrganismConstructionCode();
    occ.set_code(std::vector<OCCInstruction>{OCCInstruction::SetBlockMouth});

    editor_organism = Organism(editor_width / 2,
                               editor_height / 2,
                               Rotation::UP,
                               anatomy,
                               brain, occ,
                               sp,
                               bp, occp, occl,
                               1, 0.05, 0.1, false);

    resize_editing_grid(width, height);
    resize_image();

    initialize_gui();
    reset_scale_view();
    clear_occ();
    load_occ();

    actual_cursor.setParent(this);
    actual_cursor.setGeometry(50, 50, 5, 5);
    actual_cursor.setStyleSheet("background-color:red;");
    actual_cursor.hide();
}

void OrganismEditor::update_cell_count_label() {
    ui.label_cell_count->setText(QString::fromStdString("Block count: " + std::to_string(editor_organism.anatomy.organism_blocks.size())));
}

void OrganismEditor::update_gui() {
    ui.le_move_range           ->setText(QString::fromStdString(std::to_string(editor_organism.move_range)));
    ui.le_anatomy_mutation_rate->setText(QString::fromStdString(std::to_string(editor_organism.anatomy_mutation_rate)));
    ui.le_grid_width           ->setText(QString::fromStdString(std::to_string(editor_width)));
    ui.le_grid_height          ->setText(QString::fromStdString(std::to_string(editor_height)));
    ui.le_brain_mutation_rate  ->setText(QString::fromStdString(std::to_string(editor_organism.brain_mutation_rate)));
    update_cell_count_label();
}

void OrganismEditor::initialize_gui() {
    update_gui();

    for (int i = 1; i < NUM_WORLD_BLOCKS; i++) {
        auto & observation = BLOCK_NAMES[i];
        auto * horizontal_layout = new QHBoxLayout{};
        auto * b_group = new QButtonGroup(this);
        auto * weight_le = new QLineEdit(ui.widget_2);

        horizontal_layout->addWidget(new QLabel(QString::fromStdString(observation + " "), ui.widget_2));
        for (auto & decision: decisions) {
            auto * cb = new QCheckBox(QString::fromStdString(decision), ui.widget_2);
            connect(cb, &QCheckBox::clicked, [&, decision, observation](){brain_cb_chooser(observation, decision);});

            horizontal_layout->addWidget(cb);
            b_group->addButton(cb);

            brain_checkboxes[observation][decision] = cb;
        }
        connect(weight_le, &QLineEdit::returnPressed, [&, observation, weight_le](){brain_weight_chooser(observation, weight_le);});
        horizontal_layout->addWidget(weight_le);
        brain_line_edits[observation] = weight_le;

        ui.brain_vertical_layout->addItem(horizontal_layout);
    }

    mapped_decisions_s_to_type["Do Nothing"] = SimpleDecision::DoNothing;
    mapped_decisions_s_to_type["Go Away"]    = SimpleDecision::GoAway;
    mapped_decisions_s_to_type["Go Towards"] = SimpleDecision::GoTowards;

    for (int i = 1; i < NUM_WORLD_BLOCKS; i++) {mapped_block_types_s_to_type[BLOCK_NAMES[i]] = BlockTypes(i);}

    for (auto & pair: mapped_decisions_s_to_type)   {mapped_decisions_type_to_s  [pair.second] = pair.first;}
    for (auto & pair: mapped_block_types_s_to_type) {mapped_block_types_type_to_s[pair.second] = pair.first;}

    update_brain_state();
}

void OrganismEditor::brain_cb_chooser(std::string observation, std::string action) {
    auto observation_type = mapped_block_types_s_to_type[observation];
    auto decision = mapped_decisions_s_to_type[action];

    SimpleDecision * brain_decision = &editor_organism.brain.simple_action_table.da[int(observation_type)];

    switch (decision) {
        case SimpleDecision::DoNothing: *brain_decision = SimpleDecision::DoNothing;break;
        case SimpleDecision::GoAway:    *brain_decision = SimpleDecision::GoAway;   break;
        case SimpleDecision::GoTowards: *brain_decision = SimpleDecision::GoTowards;break;
    }
}

void OrganismEditor::brain_weight_chooser(std::string observation, QLineEdit *le) {
    auto observation_type = mapped_block_types_s_to_type[observation];

    float * weight = &editor_organism.brain.weighted_action_table.da[int(observation_type)];

    le_slot_lower_upper_bound<float>(*weight, *weight, "float", le, -1., "-1", 1., "1");
}

void OrganismEditor::update_brain_state() {
    update_brain_checkboxes();
    update_brain_line_edits();
}

void OrganismEditor::update_brain_line_edits() {
    for (int i = 1; i < NUM_WORLD_BLOCKS; i++) {
        const auto & name = BLOCK_NAMES[i];

        brain_line_edits[name]->setText(QString::fromStdString(std::to_string(editor_organism.brain.weighted_action_table.da[i])));
    }
}

void OrganismEditor::update_brain_checkboxes() {
    for (int i = 1; i < NUM_WORLD_BLOCKS; i++) {
        const auto & name = BLOCK_NAMES[i];

        brain_checkboxes[name][mapped_decisions_type_to_s[editor_organism.brain.simple_action_table.da[i]]]->setChecked(true);
    }
}

void OrganismEditor::closeEvent(QCloseEvent * event) {
    parent_ui->tb_open_organism_editor->setChecked(false);
    QWidget::closeEvent(event);
}

void OrganismEditor::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
}

void OrganismEditor::wheelEvent(QWheelEvent *event) {
    if (ui.editor_graphicsView->underMouse()) {
        if ((event->angleDelta().x() + event->angleDelta().y()) > 0) {
            scaling_zoom /= scaling_coefficient;
            if (scaling_zoom <= 0.001) {scaling_zoom *= scaling_coefficient;}
        } else {
            scaling_zoom *= scaling_coefficient;
            if (scaling_zoom >= 10) {scaling_zoom /= scaling_coefficient;}
        }
        create_image();
    }
}

void OrganismEditor::move_center(int delta_x, int delta_y) {
    center_x -= delta_x * scaling_zoom;
    center_y -= delta_y * scaling_zoom;
    create_image();
}

void OrganismEditor::reset_scale_view() {
    center_x = (float)editor_width/2;
    center_y = (float)editor_height/2;

    float exp;
    if (editor_width < editor_height) {
        exp = log((float) (editor_height+4) / (float) ui.editor_graphicsView->viewport()->height()) / log(scaling_coefficient);
    } else {
        exp = log((float) (editor_width+ 4) / (float) ui.editor_graphicsView->viewport()->width()) / log(scaling_coefficient);
    }

    scaling_zoom = pow(scaling_coefficient, exp);

    finalize_chosen_organism();
}

void OrganismEditor::resize_editing_grid(int width, int height) {
    editor_width = width;
    editor_height = height;
    edit_grid.clear();
    edit_grid.resize(width * height, BaseGridBlock{});

    editor_organism.x = editor_width / 2;
    editor_organism.y = editor_height / 2;

    int x = editor_organism.x;
    int y = editor_organism.y;

    for (int i = 0; i < editor_organism.anatomy.organism_blocks.size(); i++) {
        auto & block = editor_organism.anatomy.organism_blocks[i];
        const auto pos = block.get_pos(Rotation::UP);

        if (pos.x + x >= editor_width  || pos.x + x < 0 ||
            pos.y + y >= editor_height || pos.y + y < 0) {
            editor_organism.anatomy.organism_blocks.erase(editor_organism.anatomy.organism_blocks.begin() + i);
            i--;
        }
    }

    editor_organism.anatomy.set_many_blocks(editor_organism.anatomy.organism_blocks);

    place_organism_on_a_grid();
}

void OrganismEditor::resize_image() {
    edit_image.clear();
    edit_image.resize(4 * ui.editor_graphicsView->viewport()->width() * ui.editor_graphicsView->viewport()->height(), 255);
}

void OrganismEditor::create_image() {
    place_organism_on_a_grid();
    update_cell_count_label();

    resize_image();

    auto image_width = ui.editor_graphicsView->viewport()->width();
    auto image_height = ui.editor_graphicsView->viewport()->height();

    int scaled_width = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and y coordinates on simulation grid
    auto start_x = int(center_x-(scaled_width / 2));
    auto end_x = int(center_x+(scaled_width / 2));

    auto start_y = int(center_y-(scaled_height / 2));
    auto end_y = int(center_y+(scaled_height / 2));

    std::vector<int> lin_width;
    std::vector<int> lin_height;
    std::vector<int> truncated_lin_width;
    std::vector<int> truncated_lin_height;

    ImageCreation::calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    ImageCreation::calculate_truncated_linspace(image_width, image_height, lin_width, lin_height, truncated_lin_width, truncated_lin_height);

#ifdef __CUDA_USED__
    void * cuda_i_creator = &cuda_image_creator;
#else
    void * cuda_i_creator = nullptr;
#endif

    ImageCreation::create_image(lin_width, lin_height, editor_width, editor_height, *color_container, textures,
                                image_width, image_height, edit_image, edit_grid, use_cuda, cuda_is_available,
                                cuda_i_creator, truncated_lin_width, truncated_lin_height, false, 1);

    pixmap_item.setPixmap(QPixmap::fromImage(QImage(edit_image.data(), image_width, image_height, QImage::Format_RGB32)));
}

void OrganismEditor::clear_grid() {
    for (auto & block: edit_grid) {
        block.type = BlockTypes::EmptyBlock;
    }
}

void OrganismEditor::place_organism_on_a_grid() {
    clear_grid();
    if (check_edit_area()) {resize_editing_grid(new_editor_width, new_editor_height);}

    for (auto & block: editor_organism.anatomy.organism_blocks) {
        auto x = editor_organism.x + block.get_pos(Rotation::UP).x;
        auto y = editor_organism.y + block.get_pos(Rotation::UP).y;
        edit_grid[x + y * editor_width].type = block.type;
        edit_grid[x + y * editor_width].rotation = block.rotation;
    }
    finalize_chosen_organism();
}

Vector2<int> OrganismEditor::calculate_cursor_pos_on_grid(int x, int y) {
    x -= (ui.editor_graphicsView->x() + 6 + 1);
    y -= (ui.editor_graphicsView->y() + 6 + 1);

//    std::cout << (x - ui.editor_graphicsView->viewport()->width() /2.)*scaling_zoom + center_x + 0.75 << " " <<
//                 (y - ui.editor_graphicsView->viewport()->height()/2.)*scaling_zoom + center_y + 0.75 << "\n";

    //TODO for some reason it becomes less accurate when coordinates go away from 0,0
    auto c_pos = Vector2<int>{};
    c_pos.x = static_cast<int>((x - ui.editor_graphicsView->viewport()->width() / 2.) * scaling_zoom + center_x);
    c_pos.y = static_cast<int>((y - ui.editor_graphicsView->viewport()->height() / 2.) * scaling_zoom + center_y);
    return c_pos;
}

void OrganismEditor::finalize_chosen_organism() {
    chosen_organism->copy_organism(editor_organism);
    chosen_organism->rotation = choosen_rotation;
}

void OrganismEditor::load_chosen_organism() {
    editor_organism.copy_organism(*chosen_organism);

    editor_organism.x = editor_width / 2;
    editor_organism.y = editor_height / 2;

    if (check_edit_area()) {resize_editing_grid(new_editor_width, new_editor_height);}

    create_image();
    update_gui();
    clear_occ();
    load_occ();
    update_brain_state();
}

bool OrganismEditor::check_edit_area() {
    Vector2 min{0, 0};
    Vector2 max{0, 0};

    for (auto & block: editor_organism.anatomy.organism_blocks) {
        if (block.relative_x < min.x) {min.x = block.relative_x;}
        if (block.relative_y < min.y) {min.y = block.relative_y;}
        if (block.relative_x > max.x) {max.x = block.relative_x;}
        if (block.relative_y > max.y) {max.y = block.relative_y;}
    }

    if (std::max(abs(min.x), max.x) * 2 + 1 > new_editor_width)  { new_editor_width = std::max(abs(min.x), max.x) * 2 + 1;  return true;}
    if (std::max(abs(min.y), max.y) * 2 + 1 > new_editor_height) { new_editor_height = std::max(abs(min.y), max.y) * 2 + 1; return true;}

    return false;
}

void OrganismEditor::occ_mode(bool state) {
    if (state) {
        ui.rb_mouth->hide();
        ui.rb_producer->hide();
        ui.rb_mover->hide();
        ui.rb_killer->hide();
        ui.rb_armor->hide();
        ui.rb_eye->hide();

        ui.cmb_block_rotation->hide();
        ui.label_7->hide();

        change_disabled = true;

        b_reset_organism_slot();

        auto & occ = editor_organism.occ;
        occ.get_code_ref().clear();
        occ.get_code_ref().emplace_back(OCCInstruction::SetBlockMouth);

        ui.rb_edit_occ->show();
        ui.b_save_organism->hide();
        ui.b_load_organism->hide();
        ui.label_occ_count->show();
        clear_occ();
    } else {
        ui.rb_mouth->show();
        ui.rb_producer->show();
        ui.rb_mover->show();
        ui.rb_killer->show();
        ui.rb_armor->show();
        ui.rb_eye->show();

        ui.cmb_block_rotation->show();
        ui.label_7->show();

        change_disabled = false;

        auto & occ = editor_organism.occ;
        occ.get_code_ref().clear();

        ui.rb_edit_occ->hide();
        ui.b_save_organism->show();
        ui.b_load_organism->show();
        ui.label_occ_count->hide();
        clear_occ();
    }
}

void OrganismEditor::clear_occ() {
    ui.te_occ_edit_window->setPlainText(QString(""));
}

void OrganismEditor::load_occ() {
    if (editor_organism.occ.get_code_const_ref().empty()) { return;}

    ui.te_occ_edit_window->setPlainText(QString::fromStdString(OCCTranspiler::convert_to_text_code(editor_organism.occ.get_code_const_ref(), short_instructions)));
    ui.label_occ_count->setText(QString::fromStdString("OCC instruction count: " + std::to_string(editor_organism.occ.get_code_const_ref().size())));
}

void OrganismEditor::update_brain_edit_visibility(bool weighted_edits_visible) {
    if (weighted_edits_visible) {
        for (auto & first: brain_checkboxes) {
            for (auto & second: first.second) {
                second.second->hide();
            }
        }

        for (auto & first: brain_line_edits) {
            first.second->show();
        }
    } else {
        for (auto & first: brain_checkboxes) {
            for (auto & second: first.second) {
                second.second->show();
            }
        }

        for (auto & first: brain_line_edits) {
            first.second->hide();
        }
    }
}

