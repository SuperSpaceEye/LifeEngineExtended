// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 13.06.22.
//

#include "MainWindow.h"

//TODO increment every time saving logic changes
uint32_t SAVE_VERSION = 3;

void MainWindow::write_data(std::ofstream &os) {
    write_version(os);
    write_simulation_parameters(os);
    write_organisms_block_parameters(os);
    write_data_container_data(os);
    write_simulation_grid(os);
    write_organisms(os);
}

void MainWindow::write_version(std::ofstream &os) {
    os.write((char*)&SAVE_VERSION, sizeof(uint32_t));
}

void MainWindow::write_simulation_parameters(std::ofstream& os) {
    os.write((char*)&sp, sizeof(SimulationParameters));
}

void MainWindow::write_organisms_block_parameters(std::ofstream& os) {
    os.write((char*)&bp, sizeof(OrganismBlockParameters));
}

void MainWindow::write_data_container_data(std::ofstream& os) {
    os.write((char*)&edc.total_engine_ticks, sizeof(uint32_t));
    os.write((char*)&edc.simulation_width,   sizeof(uint32_t));
    os.write((char*)&edc.simulation_height,  sizeof(uint32_t));
}
//    void MainWindow::write_color_container(){}
void MainWindow::write_simulation_grid(std::ofstream& os) {
    for (auto & col: edc.CPU_simulation_grid) {
        os.write((char*)&col[0], sizeof(AtomicGridBlock)*col.size());
    }
}
void MainWindow::write_organisms(std::ofstream& os) {
    uint32_t size = edc.organisms.size();
    os.write((char*)&size, sizeof(uint32_t));
    for (auto & organism: edc.organisms) {
        write_organism_data(os,    organism);
        write_organism_brain(os,   organism->brain.get());
        write_organism_anatomy(os, organism->anatomy.get());
    }
}

void MainWindow::write_organism_data(std::ofstream& os, Organism * organism) {
    OrganismData data{};
    data.x                       = organism->x;
    data.y                       = organism->y;
    data.life_points             = organism->life_points;
    data.damage                  = organism->damage;
    data.max_lifetime            = organism->max_lifetime;
    data.lifetime                = organism->lifetime;
    data.anatomy_mutation_rate   = organism->anatomy_mutation_rate;
    data.brain_mutation_rate     = organism->brain_mutation_rate;
    data.food_collected          = organism->food_collected;
    data.food_needed             = organism->food_needed;
    data.move_range              = organism->move_range;
    data.rotation                = organism->rotation;
    data.move_counter            = organism->move_counter;
    data.max_decision_lifetime   = organism->max_decision_lifetime;
    data.max_do_nothing_lifetime = organism->max_do_nothing_lifetime;
    data.last_decision           = organism->last_decision;

    os.write((char*)&data, sizeof(OrganismData));
}

void MainWindow::write_organism_brain(std::ofstream& os, Brain * brain) {
    os.write((char*)brain, sizeof(Brain));
}

//TODO do i need to save spaces?

void MainWindow::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = anatomy->_organism_blocks.size();
    uint32_t producing_space_size                = anatomy->_producing_space.size();
    uint32_t eating_space_size                   = anatomy->_eating_space.size();
    uint32_t killing_space_size                  = anatomy->_killing_space.size();

    os.write((char*)&organism_blocks_size,                sizeof(uint32_t));
    os.write((char*)&producing_space_size,                sizeof(uint32_t));
    os.write((char*)&eating_space_size,                   sizeof(uint32_t));
    os.write((char*)&killing_space_size,                  sizeof(uint32_t));

    os.write((char*)&anatomy->_mouth_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_producer_blocks, sizeof(int32_t));
    os.write((char*)&anatomy->_mover_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_killer_blocks,   sizeof(int32_t));
    os.write((char*)&anatomy->_armor_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_eye_blocks,      sizeof(int32_t));

    os.write((char*)&anatomy->_organism_blocks[0],                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    os.write((char*)&anatomy->_eating_space[0],                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    os.write((char*)&anatomy->_killing_space[0],                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());

    for (auto & space: anatomy->_producing_space) {
        auto space_size = space.size();
        os.write((char*)&space_size, sizeof(uint32_t));
        os.write((char*)&space[0], sizeof(SerializedAdjacentSpaceContainer) * space_size);
    }
}

void MainWindow::recover_state(const SimulationParameters &recovery_sp, const OrganismBlockParameters &recovery_bp,
                               uint32_t recovery_simulation_width, uint32_t recovery_simulation_height) {
    sp = recovery_sp;
    bp = recovery_bp;
    edc.simulation_width = recovery_simulation_width;
    edc.simulation_height = recovery_simulation_height;
    new_simulation_width = recovery_simulation_width;
    new_simulation_height = recovery_simulation_height;

    reset_world();
}

void MainWindow::read_data(std::ifstream &is) {
    //If save version is incompatible
    if (!read_version(is)) {
        display_message("Save version is incompatible with current program version.");
        return;
    }

    partial_clear_world();
    make_border_walls();

    SimulationParameters recovery_sp = sp;
    OrganismBlockParameters recovery_bp = bp;
    uint32_t recovery_simulation_width = edc.simulation_width;
    uint32_t recovery_simulation_height = edc.simulation_height;

    try {
        read_simulation_parameters(is);
        read_organisms_block_parameters(is);
        if (read_data_container_data(is)) {
            recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        }
        read_simulation_grid(is);
        if (read_organisms(is)) {
            recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        }
    } catch (std::string & e1) { recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);}

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

bool MainWindow::read_version(std::ifstream &is) {
    int save_version;
    is.read((char*)&save_version, sizeof(int));
    return save_version == SAVE_VERSION;
}

void MainWindow::read_simulation_parameters(std::ifstream& is) {
    is.read((char*)&sp, sizeof(SimulationParameters));
}

void MainWindow::read_organisms_block_parameters(std::ifstream& is) {
    is.read((char*)&bp, sizeof(OrganismBlockParameters));
}

bool MainWindow::read_data_container_data(std::ifstream& is) {
    uint32_t sim_width;
    uint32_t sim_height;

    is.read((char*)&edc.loaded_engine_ticks, sizeof(uint32_t));
    is.read((char*)&sim_width,    sizeof(uint32_t));
    is.read((char*)&sim_height,   sizeof(uint32_t));

    if (sim_width > max_loaded_world_side) {
        if (!display_dialog_message("The loaded side of a simulation width is " + std::to_string(sim_width) + ". Continue?", false)) {
            return true;
        }
    }

    if (sim_height > max_loaded_world_side) {
        if (!display_dialog_message("The loaded side of a simulation height is " + std::to_string(sim_height) + ". Continue?", false)) {
            return true;
        }
    }

    edc.simulation_width  = sim_width;
    edc.simulation_height = sim_height;

    new_simulation_width = edc.simulation_width;
    new_simulation_height = edc.simulation_height;
    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
    update_simulation_size_label();
    return false;
}
//    void MainWindow::read_color_container(){}

//TODO only save food/walls ?
void MainWindow::read_simulation_grid(std::ifstream& is) {
    disable_warnings = true;
    resize_simulation_grid();
    disable_warnings = false;

    for (auto & col: edc.CPU_simulation_grid) {
        is.read((char*)&col[0], sizeof(AtomicGridBlock)*col.size());
    }

    for (auto & column: edc.CPU_simulation_grid) {
        for (auto &block: column) {
            if (block.type == BlockTypes::WallBlock ||
                block.type == BlockTypes::EmptyBlock ||
                block.type == BlockTypes::FoodBlock) { continue; }
            block = AtomicGridBlock();
        }
    }
}

//TODO save child patterns?
bool MainWindow::read_organisms(std::ifstream& is) {
    uint32_t num_organisms;
    is.read((char*)&num_organisms, sizeof(uint32_t));

    if (num_organisms > max_loaded_num_organisms) {
        if (!display_dialog_message("The loaded number of organisms is " + std::to_string(num_organisms) +
        ". The save file may be corrupted and could crash your computer, continue?", false)) {
            return true;
        }
    }

    edc.organisms.reserve(num_organisms);
    for (int i = 0; i < num_organisms; i++) {
        OrganismData data{};
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        read_organism_data(is, data);
        read_organism_brain(is, brain.get());
        read_organism_anatomy(is, anatomy.get());

        auto * organism = new Organism(data.x,
                                       data.y,
                                       data.rotation,
                                       anatomy,
                                       brain,
                                       &sp,
                                       &bp,
                                       data.move_range,
                                       data.anatomy_mutation_rate,
                                       data.brain_mutation_rate);

        organism->bp                      = &bp;
        organism->sp                      = &sp;
        organism->child_pattern           = nullptr;
        organism->life_points             = data.life_points;
        organism->damage                  = data.damage;
        organism->max_lifetime            = data.max_lifetime;
        organism->lifetime                = data.lifetime;
        organism->food_collected          = data.food_collected;
        organism->food_needed             = data.food_needed;
        organism->move_counter            = data.move_counter;
        organism->max_decision_lifetime   = data.max_decision_lifetime;
        organism->max_do_nothing_lifetime = data.max_do_nothing_lifetime;
        organism->last_decision           = data.last_decision;

        edc.organisms.emplace_back(organism);
        SimulationEngineSingleThread::place_organism(&edc, organism);
    }

    return false;
}

void MainWindow::read_organism_data(std::ifstream& is, OrganismData & data) {
    is.read((char*)&data, sizeof(OrganismData));
}

void MainWindow::read_organism_brain(std::ifstream& is, Brain * brain) {
    is.read((char*)brain, sizeof(Brain));
}

void MainWindow::read_organism_anatomy(std::ifstream& is, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = 0;
    uint32_t producing_space_size                = 0;
    uint32_t eating_space_size                   = 0;
    uint32_t killing_space_size                  = 0;

    is.read((char*)&organism_blocks_size,                sizeof(uint32_t));
    is.read((char*)&producing_space_size,                sizeof(uint32_t));
    is.read((char*)&eating_space_size,                   sizeof(uint32_t));
    is.read((char*)&killing_space_size,                  sizeof(uint32_t));

    anatomy->_organism_blocks.resize(organism_blocks_size);
    anatomy->_producing_space.resize(producing_space_size);
    anatomy->_eating_space   .resize(eating_space_size);
    anatomy->_killing_space  .resize(killing_space_size);

    is.read((char*)&anatomy->_mouth_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_producer_blocks, sizeof(int32_t));
    is.read((char*)&anatomy->_mover_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_killer_blocks,   sizeof(int32_t));
    is.read((char*)&anatomy->_armor_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_eye_blocks,      sizeof(int32_t));

    is.read((char*)&anatomy->_organism_blocks[0],                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    is.read((char*)&anatomy->_eating_space[0],                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    is.read((char*)&anatomy->_killing_space[0],                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());

    for (auto & space: anatomy->_producing_space) {
        uint32_t space_size;
        is.read((char*)&space_size, sizeof(uint32_t));
        space.resize(space_size);
        is.read((char*)&space[0], sizeof(SerializedAdjacentSpaceContainer) * space_size);
    }
}

void MainWindow::update_table_values() {
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 3; col++) {
            BParameters *type;
            switch (static_cast<BlocksNames>(row)) {
                case BlocksNames::MouthBlock:
                    type = &bp.MouthBlock;
                    break;
                case BlocksNames::ProducerBlock:
                    type = &bp.ProducerBlock;
                    break;
                case BlocksNames::MoverBlock:
                    type = &bp.MoverBlock;
                    break;
                case BlocksNames::KillerBlock:
                    type = &bp.KillerBlock;
                    break;
                case BlocksNames::ArmorBlock:
                    type = &bp.ArmorBlock;
                    break;
                case BlocksNames::EyeBlock:
                    type = &bp.EyeBlock;
                    break;
            }

            float *value;
            switch (static_cast<ParametersNames>(col)) {
                case ParametersNames::FoodCostModifier:
                    value = &type->food_cost_modifier;
                    break;
                case ParametersNames::LifePointAmount:
                    value = &type->life_point_amount;
                    break;
                case ParametersNames::ChanceWeight:
                    value = &type->chance_weight;
                    break;
            }
            _ui.table_organism_block_parameters->item(row, col)->setText(QString::fromStdString(to_str(*value)));
        }
    }
    _ui.table_organism_block_parameters->update();
}

namespace pt = boost::property_tree;

//TODO https://github.com/Tencent/rapidjson ?

//https://www.cochoy.fr/boost-property-tree/
void MainWindow::read_json_data(const std::string &path) {
    partial_clear_world();
    make_border_walls();

    pt::ptree root;
    pt::read_json(path, root);

    json_read_grid_data(root);

    json_read_simulation_parameters(root);

    json_read_organism_data(root);

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

void MainWindow::json_read_grid_data(boost::property_tree::ptree &root) {
    edc.simulation_height  = root.get<int>("num_rows") + 2;
    edc.simulation_width   = root.get<int>("num_cols") + 2;

    new_simulation_width = edc.simulation_width;
    new_simulation_height = edc.simulation_height;

    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    update_simulation_size_label();

    resize_simulation_grid();
    make_border_walls();
    disable_warnings = false;

    edc.loaded_engine_ticks = root.get<int>("total_ticks");

    for (auto & pair: root.get_child("grid.food")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        edc.CPU_simulation_grid[x][y].type = BlockTypes::FoodBlock;
    }

    for (auto & pair: root.get_child("grid.walls")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        edc.CPU_simulation_grid[x][y].type = BlockTypes::WallBlock;
    }
}

void MainWindow::json_read_simulation_parameters(const boost::property_tree::ptree &root) {
    sp.lifespan_multiplier               = root.get<int>("controls.lifespanMultiplier");
    sp.food_production_probability       = float(root.get<int>("controls.foodProdProb")) / 100;
    sp.use_anatomy_evolved_mutation_rate = !root.get<bool>("controls.useGlobalMutability");
    sp.global_anatomy_mutation_rate      = float(root.get<int>("controls.globalMutability")) / 100;
    sp.add_cell                          = root.get<int>("controls.addProb");
    sp.change_cell                       = root.get<int>("controls.changeProb");
    sp.remove_cell                       = root.get<int>("controls.removeProb");
    sp.runtime_rotation_enabled          = root.get<bool>("controls.rotationEnabled");
    sp.food_blocks_reproduction          = root.get<bool>("controls.foodBlocksReproduction");
    sp.movers_can_produce_food           = root.get<bool>("controls.moversCanProduce");
    sp.on_touch_kill                     = root.get<bool>("controls.instaKill");
    sp.look_range                        = root.get<int>("controls.lookRange");
    sp.extra_mover_reproductive_cost     = root.get<int>("controls.extraMoverFoodCost");
}

void MainWindow::json_read_organism_data(boost::property_tree::ptree &root) {
    for (auto & organism: root.get_child("organisms")) {
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        int y              = organism.second.get<int>("r")+1;
        int x              = organism.second.get<int>("c")+1;
        int lifetime       = organism.second.get<int>("lifetime");
        int food_collected = organism.second.get<int>("food_collected");
        int rotation       = organism.second.get<int>("rotation");
        int move_range     = organism.second.get<int>("move_range");
        float mutability   = float(organism.second.get<float>("mutability"))/100;
        int damage         = organism.second.get<int>("damage");
        bool is_mover      = organism.second.get<bool>("anatomy.is_mover");
        bool has_eyes      = organism.second.get<bool>("anatomy.has_eyes");

        auto block_data = std::vector<SerializedOrganismBlockContainer>{};

        for (auto & cell: organism.second.get_child("anatomy.cells")) {
            int l_y = cell.second.get<int>("loc_row");
            int l_x = cell.second.get<int>("loc_col");
            auto state = cell.second.get<std::string>("state.name");

            Rotation _rotation = Rotation::UP;
            BlockTypes type = BlockTypes::ProducerBlock;

            if        (state == "producer") {
                type = BlockTypes::ProducerBlock;
            } else if (state == "mouth") {
                type = BlockTypes::MouthBlock;
            } else if (state == "killer") {
                type = BlockTypes::KillerBlock;
            } else if (state == "mover") {
                type = BlockTypes::MoverBlock;
            } else if (state == "eye") {
                type = BlockTypes::EyeBlock;
                _rotation = static_cast<Rotation>(cell.second.get<int>("direction"));
            } else if (state == "armor") {
                type = BlockTypes::ArmorBlock;
            }

            block_data.emplace_back(type, _rotation, l_x, l_y);
        }
        anatomy->set_many_blocks(block_data);

        if (is_mover && has_eyes) {
            auto & table = brain->simple_action_table;
            table.FoodBlock     = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.food"));
            table.WallBlock     = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.wall"));
            table.MouthBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.mouth"));
            table.ProducerBlock = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.producer"));
            table.MoverBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.mover"));
            table.KillerBlock   = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.killer"));
            table.ArmorBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.armor"));
            table.EyeBlock      = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.eye"));
        }

        auto * new_organism = new Organism(x,
                                           y,
                                           static_cast<Rotation>(rotation),
                                           anatomy,
                                           brain,
                                           &sp,
                                           &bp,
                                           move_range,
                                           mutability);
        new_organism->lifetime = lifetime;
        new_organism->food_collected = food_collected;
        new_organism->damage = damage;
        edc.organisms.emplace_back(new_organism);
        SimulationEngineSingleThread::place_organism(&edc, new_organism);
    }
}

//#include "CustomJsonParser/json_parser.hpp"

template <typename T>
struct my_id_translator
{
    typedef T internal_type;
    typedef T external_type;

    boost::optional<T> get_value(const T &v) { return  v.substr(1, v.size() - 2) ; }
    boost::optional<T> put_value(const T &v) { return '"' + v +'"'; }
};

void MainWindow::write_json_data(const std::string &path) {
    pt::ptree root, grid, organisms, fossil_record, controls, killable_neighbors, edible_neighbors, growableNeighbors, cell, value, anatomy, cells, j_organism, food, walls, brain;

    auto info = parse_organisms_info();

    root.put("num_rows", edc.simulation_height - 2);
    root.put("num_cols", edc.simulation_width - 2);
    root.put("total_mutability", static_cast<int>(info.total_total_mutation_rate*100));
    root.put("largest_cell_count", 0);
    root.put("reset_count", 0);
    root.put("total_ticks", edc.total_engine_ticks);
    root.put("data_update_rate", 100);

    json_write_grid(grid, cell, food, walls);
    json_write_organisms(organisms, cell, anatomy, cells, j_organism, brain);
    json_write_fossil_record(fossil_record);
    json_write_controls(controls, killable_neighbors, edible_neighbors, growableNeighbors, cell, value);

    root.put_child("grid", grid);
    root.put_child("organisms", organisms);
    root.put_child("fossil_record", fossil_record);
    root.put_child("controls", controls);

//    std::stringstream ss;
//    pt::write_json(ss, root);
//    std::string my_string_to_send_somewhere_else = ss.str();

//    std::cout << my_string_to_send_somewhere_else << std::endl;

    std::fstream file;
    file.open(path, std::ios_base::out);
    pt::write_json(file, root);
    file.close();
}

void MainWindow::json_write_grid(boost::property_tree::ptree &grid, boost::property_tree::ptree &cell,
                                 boost::property_tree::ptree &food, boost::property_tree::ptree &walls) {
    grid.put("cols", edc.simulation_width - 2);
    grid.put("rows", edc.simulation_height - 2);

    bool no_food = true;
    bool no_wall = true;

    for (int x = 1; x < edc.simulation_width - 1; x++) {
        for (int y = 1; y < edc.simulation_height - 1; y++) {
            if (edc.CPU_simulation_grid[x][y].type != BlockTypes::WallBlock &&
                edc.CPU_simulation_grid[x][y].type != BlockTypes::FoodBlock) {continue;}
            cell = pt::ptree{};

            cell.put("c", x-1);
            cell.put("r", y-1);
            if (edc.CPU_simulation_grid[x][y].type == BlockTypes::FoodBlock) {
                food.push_back(std::make_pair("", cell));
                no_food = false;
            } else {
                walls.push_back(std::make_pair("", cell));
                no_wall = false;
            }
        }
    }

    cell = pt::ptree{};

    if (no_food) {
        walls.push_back(std::make_pair("", cell));
    }

    if (no_wall) {
        walls.push_back(std::make_pair("", cell));
    }

    grid.put_child("food", food);
    grid.put_child("walls", walls);
}

void MainWindow::json_write_organisms(boost::property_tree::ptree &organisms, boost::property_tree::ptree &cell,
                                      boost::property_tree::ptree &anatomy, boost::property_tree::ptree &cells,
                                      boost::property_tree::ptree &j_organism, boost::property_tree::ptree &brain) {
    for (auto & organism: edc.organisms) {
        j_organism = pt::ptree{};
        anatomy = pt::ptree{};
        cells = pt::ptree{};
        brain = pt::ptree{};

        j_organism.put("c", organism->x-1);
        j_organism.put("r", organism->y-1);
        j_organism.put("lifetime", organism->lifetime);
        j_organism.put("food_collected", static_cast<int>(organism->food_collected));
        j_organism.put("living", true);
        j_organism.put("direction", 2);
        j_organism.put("rotation", static_cast<int>(organism->rotation));
        j_organism.put("can_rotate", sp.runtime_rotation_enabled);
        j_organism.put("move_count", 0);
        j_organism.put("move_range", organism->move_range);
        j_organism.put("ignore_brain_for", 0);
        j_organism.put("mutability", static_cast<int>(organism->anatomy_mutation_rate*100));
        j_organism.put("damage", organism->damage);

        anatomy.put("birth_distance", 6);
        anatomy.put("is_producer", static_cast<bool>(organism->anatomy->_producer_blocks));
        anatomy.put("is_mover", static_cast<bool>(organism->anatomy->_mover_blocks));
        anatomy.put("has_eyes", static_cast<bool>(organism->anatomy->_eye_blocks));

        for (auto & block: organism->anatomy->_organism_blocks) {
            cell = pt::ptree{};
            std::string state_name;

            cell.put("loc_col", block.relative_x);
            cell.put("loc_row", block.relative_y);

            switch (block.type) {
                case BlockTypes::MouthBlock: state_name    = "mouth";    break;
                case BlockTypes::ProducerBlock: state_name = "producer"; break;
                case BlockTypes::MoverBlock: state_name    = "mover";    break;
                case BlockTypes::KillerBlock: state_name   = "killer";   break;
                case BlockTypes::ArmorBlock: state_name    = "armor";    break;
                case BlockTypes::EyeBlock: state_name      = "eye";      break;
                default: state_name = "producer";
            }

            if (block.type == BlockTypes::EyeBlock) {
                cell.put("direction", static_cast<int>(block.rotation));
            }

            cell.put("state.name", state_name, my_id_translator<std::string>());

            cells.push_back(std::make_pair("", cell));
        }

        anatomy.put_child("cells", cells);

        j_organism.put_child("anatomy", anatomy);

        auto & table = organism->brain->simple_action_table;

        brain.put("decisions.empty", 0);
        brain.put("decisions.food",     static_cast<int>(table.FoodBlock));
        brain.put("decisions.wall",     static_cast<int>(table.WallBlock));
        brain.put("decisions.mouth",    static_cast<int>(table.MouthBlock));
        brain.put("decisions.producer", static_cast<int>(table.ProducerBlock));
        brain.put("decisions.mover",    static_cast<int>(table.MoverBlock));
        brain.put("decisions.killer",   static_cast<int>(table.KillerBlock));
        brain.put("decisions.armor",    static_cast<int>(table.ArmorBlock));
        brain.put("decisions.eye",      static_cast<int>(table.EyeBlock));

        j_organism.put_child("brain", brain);

        j_organism.put("species_name", "0000000000", my_id_translator<std::string>());

        organisms.push_back(std::make_pair("", j_organism));
    }
}

void MainWindow::json_write_fossil_record(boost::property_tree::ptree &fossil_record) const {
    fossil_record.put("min_discard", 10);
    fossil_record.put("record_size_limit", 500);
    fossil_record.put("records", "{}");
    fossil_record.put("species", "{}");
}

void
MainWindow::json_write_controls(boost::property_tree::ptree &controls, boost::property_tree::ptree &killable_neighbors,
                                boost::property_tree::ptree &edible_neighbors,
                                boost::property_tree::ptree &growableNeighbors, boost::property_tree::ptree &cell,
                                boost::property_tree::ptree &value) const {
    controls.put("lifespanMultiplier", static_cast<float>(sp.lifespan_multiplier));
    controls.put("foodProdProb", static_cast<int>(sp.food_production_probability*100));

    cell = pt::ptree{};
    value.put_value(0);
    cell.push_back(std::make_pair("", value));
    value.put_value(1);
    cell.push_back(std::make_pair("", value));

    killable_neighbors.push_back(std::make_pair("", cell));
    edible_neighbors.push_back(std::make_pair("", cell));
    growableNeighbors.push_back(std::make_pair("", cell));

    cell = pt::ptree{};
    value.put_value(0);
    cell.push_back(std::make_pair("", value));
    value.put_value(-1);
    cell.push_back(std::make_pair("", value));

    killable_neighbors.push_back(std::make_pair("", cell));
    edible_neighbors.push_back(std::make_pair("", cell));
    growableNeighbors.push_back(std::make_pair("", cell));

    cell = pt::ptree{};
    value.put_value(1);
    cell.push_back(std::make_pair("", value));
    value.put_value(0);
    cell.push_back(std::make_pair("", value));

    killable_neighbors.push_back(std::make_pair("", cell));
    edible_neighbors.push_back(std::make_pair("", cell));
    growableNeighbors.push_back(std::make_pair("", cell));

    cell = pt::ptree{};
    value.put_value(-1);
    cell.push_back(std::make_pair("", value));
    value.put_value(0);
    cell.push_back(std::make_pair("", value));

    killable_neighbors.push_back(std::make_pair("", cell));
    edible_neighbors.push_back(std::make_pair("", cell));
    growableNeighbors.push_back(std::make_pair("", cell));

    controls.put_child("killableNeighbors", killable_neighbors);
    controls.put_child("edibleNeighbors", edible_neighbors);
    controls.put_child("growableNeighbors", growableNeighbors);

    controls.put("useGlobalMutability", !sp.use_anatomy_evolved_mutation_rate);
    controls.put("globalMutability", static_cast<int>(sp.global_anatomy_mutation_rate*100));
    controls.put("addProb", sp.add_cell);
    controls.put("changeProb", sp.change_cell);
    controls.put("removeProb", sp.remove_cell);
    controls.put("rotationEnabled", sp.runtime_rotation_enabled);
    controls.put("foodBlocksReproduction", sp.food_blocks_reproduction);
    controls.put("moversCanProduce", sp.movers_can_produce_food);
    controls.put("instaKill", sp.on_touch_kill);
    controls.put("lookRange", sp.look_range);
    controls.put("foodDropProb", sp.auto_produce_n_food);
    controls.put("extraMoverFoodCost", 0);
}