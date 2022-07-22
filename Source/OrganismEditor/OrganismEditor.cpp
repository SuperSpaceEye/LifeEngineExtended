// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 25.06.22.
//

#include "OrganismEditor.h"

void OrganismEditor::init(int width, int height, Ui::MainWindow *parent_ui, ColorContainer *color_container,
                          SimulationParameters *sp, OrganismBlockParameters *bp, CursorMode * cursor_mode, Organism ** _chosen_organism) {
    _ui.setupUi(this);
    _parent_ui = parent_ui;

    editor_width  = width;
    editor_height = height;

    this->color_container = color_container;
    c_mode = cursor_mode;

    _ui.editor_graphicsView->show();
    _ui.editor_graphicsView->setEnabled(false);

    scene.addItem(&pixmap_item);
    _ui.editor_graphicsView->setScene(&scene);

    auto anatomy = std::make_shared<Anatomy>();
    anatomy->set_block(BlockTypes::MouthBlock, Rotation::UP, 0, 0);

    auto brain = std::make_shared<Brain>();

    chosen_organism = _chosen_organism;

    editor_organism = new Organism(editor_width  / 2,
                                   editor_height / 2,
                                   Rotation::UP,
                                   anatomy,
                                   brain,
                                   sp,
                                   bp,
                                   1);

    resize_editing_grid(width, height);
    resize_image();
    create_image();

    initialize_gui();
    reset_scale_view();

    actual_cursor.setParent(this);
    actual_cursor.setGeometry(50, 50, 5, 5);
    actual_cursor.setStyleSheet("background-color:red;");
    actual_cursor.hide();

//    auto timer = new QTimer(parent());
//    timer->setInterval(200);
//    connect(timer, &QTimer::timeout, [&]{update_chosen_organism(); create_image();});

}

void OrganismEditor::update_gui() {
    _ui.le_move_range->setText(QString::fromStdString(std::to_string(editor_organism->move_range)));
    _ui.le_anatomy_mutation_rate->setText(QString::fromStdString(std::to_string(editor_organism->anatomy_mutation_rate)));
    _ui.le_grid_width->setText(QString::fromStdString(std::to_string(editor_width)));
    _ui.le_grid_height->setText(QString::fromStdString(std::to_string(editor_height)));
}

void OrganismEditor::initialize_gui() {
    update_gui();

    for (auto & observation: observations) {
        auto * horizontal_layout = new QHBoxLayout{};
        auto * b_group = new QButtonGroup(this);

        horizontal_layout->addWidget(new QLabel(QString::fromStdString(observation), _ui.widget_2));
        for (auto & decision: decisions) {
            auto * cb = new QCheckBox(QString::fromStdString(decision), _ui.widget_2);
            connect(cb, &QCheckBox::clicked, [&, decision, observation](){brain_cb_chooser(observation, decision);});

            horizontal_layout->addWidget(cb);
            b_group->addButton(cb);

            brain_checkboxes[observation][decision] = cb;
        }

        _ui.brain_vertical_layout->addItem(horizontal_layout);
    }

    mapped_decisions_s_to_type["Do Nothing"] = SimpleDecision::DoNothing;
    mapped_decisions_s_to_type["Go Away"]    = SimpleDecision::GoAway;
    mapped_decisions_s_to_type["Go Towards"] = SimpleDecision::GoTowards;

    mapped_block_types_s_to_type["Mouth Cell"]    = BlockTypes::MouthBlock;
    mapped_block_types_s_to_type["Producer Cell"] = BlockTypes::ProducerBlock;
    mapped_block_types_s_to_type["Mover Cell"]    = BlockTypes::MoverBlock;
    mapped_block_types_s_to_type["Killer Cell"]   = BlockTypes::KillerBlock;
    mapped_block_types_s_to_type["Armor Cell"]    = BlockTypes::ArmorBlock;
    mapped_block_types_s_to_type["Eye Cell"]      = BlockTypes::EyeBlock;
    mapped_block_types_s_to_type["Food"]     = BlockTypes::FoodBlock;
    mapped_block_types_s_to_type["Wall"]          = BlockTypes::WallBlock;

    for (auto & pair: mapped_decisions_s_to_type)   {mapped_decisions_type_to_s  [pair.second] = pair.first;}
    for (auto & pair: mapped_block_types_s_to_type) {mapped_block_types_type_to_s[pair.second] = pair.first;}

    update_brain_checkboxes();
}

void OrganismEditor::brain_cb_chooser(std::string observation, std::string action) {
    auto observation_type = mapped_block_types_s_to_type[observation];
    auto decision = mapped_decisions_s_to_type[action];

    SimpleDecision * brain_decision;

    switch (observation_type) {
        case BlockTypes::MouthBlock:    brain_decision = &editor_organism->brain->simple_action_table.MouthBlock;    break;
        case BlockTypes::ProducerBlock: brain_decision = &editor_organism->brain->simple_action_table.ProducerBlock; break;
        case BlockTypes::MoverBlock:    brain_decision = &editor_organism->brain->simple_action_table.MoverBlock;    break;
        case BlockTypes::KillerBlock:   brain_decision = &editor_organism->brain->simple_action_table.KillerBlock;   break;
        case BlockTypes::ArmorBlock:    brain_decision = &editor_organism->brain->simple_action_table.ArmorBlock;    break;
        case BlockTypes::EyeBlock:      brain_decision = &editor_organism->brain->simple_action_table.EyeBlock;      break;
        case BlockTypes::FoodBlock:     brain_decision = &editor_organism->brain->simple_action_table.FoodBlock;     break;
        case BlockTypes::WallBlock:     brain_decision = &editor_organism->brain->simple_action_table.WallBlock;     break;
    }

    switch (decision) {
        case SimpleDecision::DoNothing: *brain_decision = SimpleDecision::DoNothing;break;
        case SimpleDecision::GoAway:    *brain_decision = SimpleDecision::GoAway;   break;
        case SimpleDecision::GoTowards: *brain_decision = SimpleDecision::GoTowards;break;
    }
}

void OrganismEditor::update_brain_checkboxes() {
    brain_checkboxes["Mouth Cell"]   [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.MouthBlock]]   ->setChecked(true);
    brain_checkboxes["Producer Cell"][mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.ProducerBlock]]->setChecked(true);
    brain_checkboxes["Mover Cell"]   [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.MoverBlock]]   ->setChecked(true);
    brain_checkboxes["Killer Cell"]  [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.KillerBlock]]  ->setChecked(true);
    brain_checkboxes["Armor Cell"]   [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.ArmorBlock]]   ->setChecked(true);
    brain_checkboxes["Eye Cell"]     [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.EyeBlock]]     ->setChecked(true);
    brain_checkboxes["Food"]         [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.FoodBlock]]    ->setChecked(true);
    brain_checkboxes["Wall"]         [mapped_decisions_type_to_s[editor_organism->brain->simple_action_table.WallBlock]]    ->setChecked(true);
}

void OrganismEditor::closeEvent(QCloseEvent * event) {
    _parent_ui->tb_open_organism_editor->setChecked(false);
    QWidget::closeEvent(event);
}

void OrganismEditor::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
}

void OrganismEditor::wheelEvent(QWheelEvent *event) {
    if (_ui.editor_graphicsView->underMouse()) {
        if (event->delta() > 0) {
            scaling_zoom /= scaling_coefficient;
        } else {
            scaling_zoom *= scaling_coefficient;
        }
        create_image();
    }
}

void OrganismEditor::move_center(int delta_x, int delta_y) {
    center_x -= delta_x * scaling_zoom;
    center_y -= delta_y * scaling_zoom;
    create_image();
//    std::this_thread::sleep_for(std::chrono::milliseconds(int(1./60*1000)));
}

void OrganismEditor::reset_scale_view() {
    center_x = (float)editor_width/2;
    center_y = (float)editor_height/2;

    float exp;
    if (editor_width < editor_height) {
        exp = log((float) (editor_height+4) / (float) _ui.editor_graphicsView->viewport()->height()) / log(scaling_coefficient);
    } else {
        exp = log((float) (editor_width+ 4) / (float) _ui.editor_graphicsView->viewport()->width())  / log(scaling_coefficient);
    }

    scaling_zoom = pow(scaling_coefficient, exp);

    finalize_chosen_organism();
    create_image();
}

void OrganismEditor::resize_editing_grid(int width, int height) {
    editor_width = width;
    editor_height = height;
    edit_grid.clear();
    edit_grid.resize(width, std::vector<EditBlock>(height, EditBlock{}));

    editor_organism->x = editor_width / 2;
    editor_organism->y = editor_height / 2;

    int x = editor_organism->x;
    int y = editor_organism->y;

    for (int i = 0; i < editor_organism->anatomy->_organism_blocks.size(); i++) {
        auto & block = editor_organism->anatomy->_organism_blocks[i];

        if (block.get_pos(Rotation::UP).x + x >= editor_width  || block.get_pos(Rotation::UP).x + x < 0 ||
            block.get_pos(Rotation::UP).y + y >= editor_height || block.get_pos(Rotation::UP).y + y < 0) {
            editor_organism->anatomy->_organism_blocks.erase(editor_organism->anatomy->_organism_blocks.begin()+i);
            i--;
        }
    }

    editor_organism->anatomy->set_many_blocks(editor_organism->anatomy->_organism_blocks);

    place_organism_on_a_grid();
}

void OrganismEditor::resize_image() {
    edit_image.clear();
    edit_image.reserve(4 * _ui.editor_graphicsView->viewport()->width() * _ui.editor_graphicsView->viewport()->height());
}

void OrganismEditor::create_image() {
    place_organism_on_a_grid();

    resize_image();
    auto image_width = _ui.editor_graphicsView->viewport()->width();
    auto image_height = _ui.editor_graphicsView->viewport()->height();

    int scaled_width = image_width * scaling_zoom;
    int scaled_height = image_height * scaling_zoom;

    // start and y coordinates on simulation grid
    auto start_x = int(center_x-(scaled_width / 2));
    auto end_x = int(center_x+(scaled_width / 2));

    auto start_y = int(center_y-(scaled_height / 2));
    auto end_y = int(center_y+(scaled_height / 2));

    std::vector<int> lin_width;
    std::vector<int> lin_height;

    calculate_linspace(lin_width, lin_height, start_x, end_x, start_y, end_y, image_width, image_height);

    complex_for_loop(lin_width, lin_height);

    pixmap_item.setPixmap(QPixmap::fromImage(QImage(edit_image.data(), image_width, image_height, QImage::Format_RGB32)));
}

void OrganismEditor::calculate_linspace(std::vector<int> & lin_width, std::vector<int> & lin_height,
                                    int start_x, int end_x, int start_y, int end_y, int image_width, int image_height) {
    lin_width  = linspace<int>(start_x, end_x, image_width);
    lin_height = linspace<int>(start_y, end_y, image_height);
}

void OrganismEditor::complex_for_loop(std::vector<int> &lin_width, std::vector<int> &lin_height) {
    //x - start, y - stop
    std::vector<Vector2<int>> width_img_boundaries;
    std::vector<Vector2<int>> height_img_boundaries;

    auto last = INT32_MIN;
    auto count = 0;
    for (int x = 0; x < lin_width.size(); x++) {
        if (last < lin_width[x]) {
            width_img_boundaries.emplace_back(count, x);
            last = lin_width[x];
            count = x;
        }
    }
    width_img_boundaries.emplace_back(count, lin_width.size());

    last = INT32_MIN;
    count = 0;
    for (int x = 0; x < lin_height.size(); x++) {
        if (last < lin_height[x]) {
            height_img_boundaries.emplace_back(count, x);
            last = lin_height[x];
            count = x;
        }
    }
    height_img_boundaries.emplace_back(count, lin_height.size());

    color pixel_color;
    //width of boundaries of an organisms

    //width bound, height bound
    for (auto &w_b: width_img_boundaries) {
        for (auto &h_b: height_img_boundaries) {
            for (int x = w_b.x; x < w_b.y; x++) {
                for (int y = h_b.x; y < h_b.y; y++) {
                    if (lin_width[x] < 0 ||
                        lin_width[x] >= editor_width ||
                        lin_height[y] < 0 ||
                        lin_height[y] >= editor_height) {
                        pixel_color = color_container->simulation_background_color;
                        set_image_pixel(x, y, pixel_color);
                        continue;
                    }

                    auto &block = edit_grid[lin_width[x]][lin_height[y]];

                    pixel_color = get_texture_color(block.type,
                                                    block.rotation,
                                                    float(x - w_b.x) / (w_b.y - w_b.x),
                                                    float(y - h_b.x) / (h_b.y - h_b.x));
                    set_image_pixel(x, y, pixel_color);
                }
            }
        }
    }
}

void OrganismEditor::set_image_pixel(int x, int y, color &color) {
    auto index = 4 * (y * _ui.editor_graphicsView->viewport()->width() + x);
    edit_image[index+2] = color.r;
    edit_image[index+1] = color.g;
    edit_image[index  ] = color.b;
}

color & OrganismEditor::get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale) {
    int x;
    int y;

    switch (type) {
        case BlockTypes::EmptyBlock :   return color_container->empty_block;
        case BlockTypes::MouthBlock:    return color_container->mouth;
        case BlockTypes::ProducerBlock: return color_container->producer;
        case BlockTypes::MoverBlock:    return color_container->mover;
        case BlockTypes::KillerBlock:   return color_container->killer;
        case BlockTypes::ArmorBlock:    return color_container->armor;
        case BlockTypes::EyeBlock:
            x = relative_x_scale * 5;
            y = relative_y_scale * 5;
            {
                switch (rotation) {
                    case Rotation::UP:
                        break;
                    case Rotation::LEFT:
                        x -= 2;
                        y -= 2;
                        std::swap(x, y);
                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::DOWN:
                        x -= 2;
                        y -= 2;
                        x = -x;
                        y = -y;
                        x += 2;
                        y += 2;
                        break;
                    case Rotation::RIGHT:
                        x -= 2;
                        y -= 2;
                        std::swap(x, y);
                        x = -x;
                        x += 2;
                        y += 2;
                        break;
                }
            }
            return textures.rawEyeTexture[x + y * 5];
        case BlockTypes::FoodBlock:     return color_container->food;
        case BlockTypes::WallBlock:     return color_container->wall;
        default: return color_container->empty_block;
    }
}

void OrganismEditor::clear_grid() {
    for (auto & row: edit_grid) {
        for (auto & block: row) {
            block.type = BlockTypes::EmptyBlock;
        }
    }
}

void OrganismEditor::place_organism_on_a_grid() {
    clear_grid();

    for (auto & block: editor_organism->anatomy->_organism_blocks) {
        auto x = editor_organism->x + block.get_pos(Rotation::UP).x;
        auto y = editor_organism->y + block.get_pos(Rotation::UP).y;
        edit_grid[x][y].type = block.type;
        edit_grid[x][y].rotation = block.rotation;
    }
    finalize_chosen_organism();
}

Vector2<int> OrganismEditor::calculate_cursor_pos_on_grid(int x, int y) {
    x -= (_ui.editor_graphicsView->x() + 6 + 1);
    y -= (_ui.editor_graphicsView->y() + 6 + 1);

//    std::cout << (x - _ui.editor_graphicsView->viewport()->width() /2.)*scaling_zoom + center_x + 0.75 << " " <<
//                 (y - _ui.editor_graphicsView->viewport()->height()/2.)*scaling_zoom + center_y + 0.75 << "\n";

    //TODO for some reason it becomes less accurate when coordinates go away from 0,0
    auto c_pos = Vector2<int>{};
    c_pos.x = static_cast<int>((x - _ui.editor_graphicsView->viewport()->width() /2.)*scaling_zoom + center_x);
    c_pos.y = static_cast<int>((y - _ui.editor_graphicsView->viewport()->height()/2.)*scaling_zoom + center_y);
    return c_pos;
}

//TODO
void OrganismEditor::finalize_chosen_organism() {
    delete *chosen_organism;
    *chosen_organism = new Organism(editor_organism);
}

//TODO
void OrganismEditor::update_chosen_organism() {
    editor_organism = new Organism(*chosen_organism);

    Vector2 min{0, 0};
    Vector2 max{0, 0};

    for (auto & block: editor_organism->anatomy->_organism_blocks) {
        if (block.relative_x < min.x) {min.x = block.relative_x;}
        if (block.relative_y < min.y) {min.y = block.relative_y;}
        if (block.relative_x > max.x) {max.x = block.relative_x;}
        if (block.relative_y > max.y) {max.y = block.relative_y;}
    }

    if (std::abs(min.x) + max.x >= new_editor_width)  {new_editor_width  = std::abs(min.x) + max.x;}
    if (std::abs(min.y) + max.y >= new_editor_height) {new_editor_height = std::abs(min.y) + max.y;}

    resize_editing_grid(new_editor_width, new_editor_height);
    create_image();
    update_gui();
    update_brain_checkboxes();
}

