// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 13.06.22.
//

#include "MainWindow.h"

//TODO increment every time saving logic changes
uint32_t SAVE_VERSION = 5;

struct WorldBlocks {
    uint32_t x;
    uint32_t y;
    BlockTypes type;
    WorldBlocks()=default;
    WorldBlocks(uint32_t x, uint32_t y, BlockTypes type): x(x), y(y), type(type) {}
};

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
    std::vector<WorldBlocks> blocks{};

    for (uint32_t x = 0; x < edc.simulation_width; x++) {
        for (uint32_t y = 0; y < edc.simulation_height; y++) {
            auto & block = edc.CPU_simulation_grid[x][y];

            switch (block.type) {
                case BlockTypes::FoodBlock:
                case BlockTypes::WallBlock:
                    blocks.emplace_back(x, y, block.type);
                default: break;
            }
        }
    }

    auto size = blocks.size();

    os.write((char*)&size, sizeof(std::size_t));
    os.write((char*)&blocks[0], sizeof(WorldBlocks)*blocks.size());
}
void MainWindow::write_organisms(std::ofstream& os) {
    uint32_t size = edc.organisms.size();
    os.write((char*)&size, sizeof(uint32_t));
    for (auto & organism: edc.organisms) {
        write_organism_brain(os,   organism->brain.get());
        write_organism_anatomy(os, organism->anatomy.get());
        write_organism_data(os,    organism);
    }
}

void MainWindow::write_organism_data(std::ofstream& os, Organism * organism) {
    os.write((char*)static_cast<OrganismData*>(organism), sizeof(OrganismData));
}

void MainWindow::write_organism_brain(std::ofstream& os, Brain * brain) {
    os.write((char*)brain, sizeof(Brain));
}

//TODO do i need to save spaces?

void MainWindow::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
    uint32_t organism_blocks_size = anatomy->_organism_blocks.size();
    uint32_t producing_space_size = anatomy->_producing_space.size();
    uint32_t eating_space_size    = anatomy->_eating_space.size();
    uint32_t killing_space_size   = anatomy->_killing_space.size();

    os.write((char*)&organism_blocks_size, sizeof(uint32_t));
    os.write((char*)&producing_space_size, sizeof(uint32_t));
    os.write((char*)&eating_space_size,    sizeof(uint32_t));
    os.write((char*)&killing_space_size,   sizeof(uint32_t));

    os.write((char*)&anatomy->_mouth_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_producer_blocks, sizeof(int32_t));
    os.write((char*)&anatomy->_mover_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_killer_blocks,   sizeof(int32_t));
    os.write((char*)&anatomy->_armor_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_eye_blocks,      sizeof(int32_t));

    os.write((char*)&anatomy->_organism_blocks[0], sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    os.write((char*)&anatomy->_eating_space[0],    sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    os.write((char*)&anatomy->_killing_space[0],   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());

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
    edc.simulation_width  = recovery_simulation_width;
    edc.simulation_height = recovery_simulation_height;
    new_simulation_width  = recovery_simulation_width;
    new_simulation_height = recovery_simulation_height;

    engine->reset_world();
    unpause_engine();
}

void MainWindow::read_data(std::ifstream &is) {
    //If save version is incompatible
    if (!read_version(is)) {
        display_message("Save version is incompatible with current program version.");
        return;
    }

    engine->partial_clear_world();
    engine->make_walls();

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
    auto flag = disable_warnings;
    disable_warnings = true;
    resize_simulation_grid();
    disable_warnings = flag;

    std::vector<WorldBlocks> blocks{};
    std::size_t size;

    is.read((char*)&size, sizeof(std::size_t));
    blocks.resize(size);

    is.read((char*)&blocks[0], sizeof(WorldBlocks)*size);

    for (auto & block: blocks) {
        edc.CPU_simulation_grid[block.x][block.y].type = block.type;
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
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        read_organism_brain(is, brain.get());
        read_organism_anatomy(is, anatomy.get());

        auto * organism = new Organism(0,
                                       0,
                                       Rotation::UP,
                                       anatomy,
                                       brain,
                                       &sp,
                                       &bp,
                                       0,
                                       0,
                                       0);

        read_organism_data(is, *static_cast<OrganismData*>(organism));

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
    uint32_t organism_blocks_size = 0;
    uint32_t producing_space_size = 0;
    uint32_t eating_space_size    = 0;
    uint32_t killing_space_size   = 0;

    is.read((char*)&organism_blocks_size, sizeof(uint32_t));
    is.read((char*)&producing_space_size, sizeof(uint32_t));
    is.read((char*)&eating_space_size,    sizeof(uint32_t));
    is.read((char*)&killing_space_size,   sizeof(uint32_t));

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

    is.read((char*)&anatomy->_organism_blocks[0], sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    is.read((char*)&anatomy->_eating_space[0],    sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    is.read((char*)&anatomy->_killing_space[0],   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());

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
                case ParametersNames::LifetimeWeight:
                    value = &type->lifetime_weight;
                    break;
            }
            _ui.table_organism_block_parameters->item(row, col)->setText(QString::fromStdString(to_str(*value)));
        }
    }
    _ui.table_organism_block_parameters->update();
}

using rapidjson::Value, rapidjson::Document, rapidjson::StringBuffer, rapidjson::Writer, rapidjson::kObjectType, rapidjson::kArrayType;

void MainWindow::read_json_data(const std::string &path) {
    std::string json;
    auto ss = std::ostringstream();
    std::ifstream file;
    file.open(path);
    if (!file.is_open()) {
        return;
    }

    ss << file.rdbuf();
    json = ss.str();

    file.close();

    Document document;
    document.Parse(json.c_str());

    if (!document.HasMember("num_rows")) {
        display_message("Failed to load world");
        return;
    }

    engine->partial_clear_world();
    engine->make_walls();

    json_read_grid_data(document);
    json_read_simulation_parameters(document);
    json_read_organisms_data(document);

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

void MainWindow::json_read_grid_data(rapidjson::Document & d) {
    edc.simulation_height = d["num_rows"].GetInt() + 2;
    edc.simulation_width  = d["num_cols"].GetInt() + 2;

    new_simulation_width = edc.simulation_width;
    new_simulation_height = edc.simulation_height;

    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    update_simulation_size_label();

    resize_simulation_grid();
    engine->make_walls();
    disable_warnings = false;

    edc.loaded_engine_ticks = d["total_ticks"].GetInt64();

    for (auto & pos: d["grid"]["food"].GetArray()) {
        int x = pos["c"].GetInt() + 1;
        int y = pos["r"].GetInt() + 1;

        edc.CPU_simulation_grid[x][y].type = BlockTypes::FoodBlock;
    }

    for (auto & pos: d["grid"]["walls"].GetArray()) {
        int x = pos["c"].GetInt() + 1;
        int y = pos["r"].GetInt() + 1;

        edc.CPU_simulation_grid[x][y].type = BlockTypes::WallBlock;
    }
}

void MainWindow::json_read_simulation_parameters(rapidjson::Document & d) {
    sp.lifespan_multiplier               = d["controls"]["lifespanMultiplier"].GetFloat();
    sp.food_production_probability       = d["controls"]["foodProdProb"].GetFloat() / 100;
    sp.use_anatomy_evolved_mutation_rate =!d["controls"]["useGlobalMutability"].GetBool();
    sp.global_anatomy_mutation_rate      = d["controls"]["globalMutability"].GetFloat() / 100;
    sp.add_cell                          = d["controls"]["addProb"].GetInt();
    sp.change_cell                       = d["controls"]["changeProb"].GetInt();
    sp.remove_cell                       = d["controls"]["removeProb"].GetInt();
    sp.runtime_rotation_enabled          = d["controls"]["rotationEnabled"].GetBool();
    sp.food_blocks_reproduction          = d["controls"]["foodBlocksReproduction"].GetBool();
    sp.movers_can_produce_food           = d["controls"]["moversCanProduce"].GetBool();
    sp.on_touch_kill                     = d["controls"]["instaKill"].GetBool();
    sp.look_range                        = d["controls"]["lookRange"].GetInt();
    sp.extra_mover_reproductive_cost     = d["controls"]["extraMoverFoodCost"].GetFloat();
}

void MainWindow::json_read_organisms_data(rapidjson::Document & d) {
    for (auto & organism: d["organisms"].GetArray()) {
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        int y                = organism["r"].GetInt()+1;
        int x                = organism["c"].GetInt()+1;
        int lifetime         = organism["lifetime"].GetInt();
        float food_collected = organism["food_collected"].GetFloat();
        int rotation         = organism["rotation"].GetInt();
        int move_range       = organism["move_range"].GetInt();
        float mutability     = organism["mutability"].GetFloat()/100;
        float damage         = organism["damage"].GetFloat();
        bool is_mover        = organism["anatomy"]["is_mover"].GetBool();
        bool has_eyes        = organism["anatomy"]["has_eyes"].GetBool();

        auto block_data = std::vector<SerializedOrganismBlockContainer>{};

        for (auto & cell: organism["anatomy"]["cells"].GetArray()) {
            int l_x = cell["loc_col"].GetInt();
            int l_y = cell["loc_row"].GetInt();
            auto state = std::string(cell["state"]["name"].GetString());

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
                _rotation = static_cast<Rotation>(cell["direction"].GetInt());
            } else if (state == "armor") {
                type = BlockTypes::ArmorBlock;
            }

            block_data.emplace_back(type, _rotation, l_x, l_y);
        }
        anatomy->set_many_blocks(block_data);

        if (is_mover && has_eyes) {
            auto & table = brain->simple_action_table;
            table.FoodBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["food"]    .GetInt());
            table.WallBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["wall"]    .GetInt());
            table.MouthBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mouth"]   .GetInt());
            table.ProducerBlock = static_cast<SimpleDecision>(organism["brain"]["decisions"]["producer"].GetInt());
            table.MoverBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mover"]   .GetInt());
            table.KillerBlock   = static_cast<SimpleDecision>(organism["brain"]["decisions"]["killer"]  .GetInt());
            table.ArmorBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["armor"]   .GetInt());
            table.EyeBlock      = static_cast<SimpleDecision>(organism["brain"]["decisions"]["eye"]     .GetInt());
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

void MainWindow::write_json_data(const std::string &path) {
    Document d;
    d.SetObject();

    auto info = parse_organisms_info();

    d.AddMember("num_rows", Value(edc.simulation_height - 2), d.GetAllocator());
    d.AddMember("num_cols", Value(edc.simulation_width - 2), d.GetAllocator());
    d.AddMember("total_mutability", Value(int(info.total_total_mutation_rate*100)), d.GetAllocator());
    d.AddMember("largest_cell_count", Value(0), d.GetAllocator());
    d.AddMember("reset_count", Value(0), d.GetAllocator());
    d.AddMember("total_ticks", Value(edc.total_engine_ticks), d.GetAllocator());
    d.AddMember("data_update_rate", Value(100), d.GetAllocator());

    json_write_grid(d);
    json_write_organisms(d);
    json_write_fossil_record(d);
    json_write_controls(d);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    std::fstream file;
    file.open(path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}

void MainWindow::json_write_grid(rapidjson::Document & d) {
    Value j_grid(kObjectType);
    j_grid.AddMember("cols", Value(edc.simulation_width  - 2), d.GetAllocator());
    j_grid.AddMember("rows", Value(edc.simulation_height - 2), d.GetAllocator());

    Value food(kArrayType);
    Value walls(kArrayType);

    for (int x = 1; x < edc.simulation_width - 1; x++) {
        for (int y = 1; y < edc.simulation_height - 1; y++) {
            if (edc.CPU_simulation_grid[x][y].type != BlockTypes::WallBlock &&
                edc.CPU_simulation_grid[x][y].type != BlockTypes::FoodBlock) {continue;}
            Value cell(kObjectType);

            cell.AddMember("c", Value(x-1), d.GetAllocator());
            cell.AddMember("r", Value(y-1), d.GetAllocator());

            if (edc.CPU_simulation_grid[x][y].type == BlockTypes::FoodBlock) {
                food.PushBack(cell, d.GetAllocator());
            } else {
                walls.PushBack(cell, d.GetAllocator());
            }
        }
    }

    j_grid.AddMember("food", food, d.GetAllocator());
    j_grid.AddMember("walls", walls, d.GetAllocator());

    d.AddMember("grid", j_grid, d.GetAllocator());
}

void MainWindow::json_write_organisms(rapidjson::Document & d) {
    Value j_organisms(kArrayType);

    for (auto & organism: edc.organisms) {
        Value j_organism(kObjectType);
        write_json_organism(d, organism, j_organism);
        j_organisms.PushBack(j_organism, d.GetAllocator());
    }
    d.AddMember("organisms", j_organisms, d.GetAllocator());
}

void MainWindow::write_json_organism(Document &d, Organism *&organism, Value &j_organism) const {
    Value j_anatomy(kObjectType);
    Value j_brain(kObjectType);
    Value cells(kArrayType);

    j_organism.AddMember("c",                Value(organism->x-1), d.GetAllocator());
    j_organism.AddMember("r",                Value(organism->y-1), d.GetAllocator());
    j_organism.AddMember("lifetime",         Value(organism->lifetime), d.GetAllocator());
    j_organism.AddMember("food_collected",   Value(organism->food_collected), d.GetAllocator());
    j_organism.AddMember("living",           Value(true), d.GetAllocator());
    j_organism.AddMember("direction",        Value(2), d.GetAllocator());
    j_organism.AddMember("rotation",         Value(static_cast<int>(organism->rotation)), d.GetAllocator());
    j_organism.AddMember("can_rotate", Value(sp.runtime_rotation_enabled), d.GetAllocator());
    j_organism.AddMember("move_count",       Value(0), d.GetAllocator());
    j_organism.AddMember("move_range",       Value(organism->move_range), d.GetAllocator());
    j_organism.AddMember("ignore_brain_for", Value(0), d.GetAllocator());
    j_organism.AddMember("mutability",       Value(organism->anatomy_mutation_rate*100), d.GetAllocator());
    j_organism.AddMember("damage",           Value(organism->damage), d.GetAllocator());

    j_anatomy.AddMember("birth_distance", Value(6), d.GetAllocator());
    j_anatomy.AddMember("is_producer",    Value(static_cast<bool>(organism->anatomy->_producer_blocks)), d.GetAllocator());
    j_anatomy.AddMember("is_mover",       Value(static_cast<bool>(organism->anatomy->_mover_blocks)), d.GetAllocator());
    j_anatomy.AddMember("has_eyes",       Value(static_cast<bool>(organism->anatomy->_eye_blocks)), d.GetAllocator());

    for (auto & block: organism->anatomy->_organism_blocks) {
        Value cell(kObjectType);
        std::string state_name;

        cell.AddMember("loc_col", Value(block.relative_x), d.GetAllocator());
        cell.AddMember("loc_row", Value(block.relative_y), d.GetAllocator());

        switch (block.type) {
            case BlockTypes::MouthBlock: state_name    = "mouth";    break;
            case BlockTypes::ProducerBlock: state_name = "producer"; break;
            case BlockTypes::MoverBlock: state_name    = "mover";    break;
            case BlockTypes::KillerBlock: state_name   = "killer";   break;
            case BlockTypes::ArmorBlock: state_name    = "armor";    break;
            case BlockTypes::EyeBlock: state_name      = "eye";      break;
//                default: state_name = "producer";
            default: continue;
        }

        if (block.type == BlockTypes::EyeBlock) {
            cell.AddMember("direction", Value(static_cast<int>(block.rotation)), d.GetAllocator());
        }

        Value state(kObjectType);
        state.AddMember("name", Value(state_name.c_str(), state_name.length(), d.GetAllocator()), d.GetAllocator());

        cell.AddMember("state", state, d.GetAllocator());

        cells.PushBack(cell, d.GetAllocator());
    }
    j_anatomy.AddMember("cells", cells, d.GetAllocator());

    j_organism.AddMember("anatomy", j_anatomy, d.GetAllocator());

    auto & table = organism->brain->simple_action_table;

    Value decisions(kObjectType);

    decisions.AddMember("empty",    Value(0), d.GetAllocator());
    decisions.AddMember("food",     Value(static_cast<int>(table.FoodBlock)),     d.GetAllocator());
    decisions.AddMember("wall",     Value(static_cast<int>(table.WallBlock)),     d.GetAllocator());
    decisions.AddMember("mouth",    Value(static_cast<int>(table.MouthBlock)),    d.GetAllocator());
    decisions.AddMember("producer", Value(static_cast<int>(table.ProducerBlock)), d.GetAllocator());
    decisions.AddMember("mover",    Value(static_cast<int>(table.MoverBlock)),    d.GetAllocator());
    decisions.AddMember("killer",   Value(static_cast<int>(table.KillerBlock)),   d.GetAllocator());
    decisions.AddMember("armor",    Value(static_cast<int>(table.ArmorBlock)),    d.GetAllocator());
    decisions.AddMember("eye",      Value(static_cast<int>(table.EyeBlock)),      d.GetAllocator());

    j_brain.AddMember("decisions", decisions, d.GetAllocator());

    j_organism.AddMember("brain", j_brain, d.GetAllocator());

    j_organism.AddMember("species_name", Value("0000000000"), d.GetAllocator());
}

void MainWindow::json_write_fossil_record(rapidjson::Document & d) {
    Value j_fossil_record(kObjectType);

    j_fossil_record.AddMember("min_discard",       Value(10), d.GetAllocator());
    j_fossil_record.AddMember("record_size_limit", Value(500), d.GetAllocator());
    j_fossil_record.AddMember("records",           Value(kObjectType), d.GetAllocator());
    j_fossil_record.AddMember("species",           Value(kObjectType), d.GetAllocator());

    d.AddMember("fossil_record", j_fossil_record, d.GetAllocator());
}

void MainWindow::json_write_controls(rapidjson::Document & d) const {
    Value j_controls(kObjectType);

    j_controls.AddMember("lifespanMultiplier", Value(sp.lifespan_multiplier), d.GetAllocator());
    j_controls.AddMember("foodProdProb", Value(sp.food_production_probability*100), d.GetAllocator());

    Value j_killable_neighbors(kArrayType);
    Value j_edible_neighbors(kArrayType);
    Value j_growableNeighbors(kArrayType);

    Value cell(kArrayType);
    cell.PushBack(Value(0), d.GetAllocator());
    cell.PushBack(Value(1), d.GetAllocator());

    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());

    cell = Value(kArrayType);
    cell.PushBack(Value(0), d.GetAllocator());
    cell.PushBack(Value(-1), d.GetAllocator());

    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());

    cell = Value(kArrayType);
    cell.PushBack(Value(1), d.GetAllocator());
    cell.PushBack(Value(0), d.GetAllocator());

    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());

    cell = Value(kArrayType);
    cell.PushBack(Value(-1), d.GetAllocator());
    cell.PushBack(Value(0), d.GetAllocator());

    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());

    j_controls.AddMember("killableNeighbors", j_killable_neighbors, d.GetAllocator());
    j_controls.AddMember("edibleNeighbors",   j_edible_neighbors,   d.GetAllocator());
    j_controls.AddMember("growableNeighbors", j_growableNeighbors,  d.GetAllocator());

    j_controls.AddMember("useGlobalMutability",    Value(!sp.use_anatomy_evolved_mutation_rate), d.GetAllocator());
    j_controls.AddMember("globalMutability",       Value(sp.global_anatomy_mutation_rate*100), d.GetAllocator());
    j_controls.AddMember("addProb",                Value(sp.add_cell), d.GetAllocator());
    j_controls.AddMember("changeProb",             Value(sp.change_cell), d.GetAllocator());
    j_controls.AddMember("removeProb",             Value(sp.remove_cell), d.GetAllocator());
    j_controls.AddMember("rotationEnabled",        Value(sp.runtime_rotation_enabled), d.GetAllocator());
    j_controls.AddMember("foodBlocksReproduction", Value(sp.food_blocks_reproduction), d.GetAllocator());
    j_controls.AddMember("moversCanProduce",       Value(sp.movers_can_produce_food), d.GetAllocator());
    j_controls.AddMember("instaKill",              Value(sp.on_touch_kill), d.GetAllocator());
    j_controls.AddMember("lookRange",              Value(sp.look_range), d.GetAllocator());
    j_controls.AddMember("foodDropProb",           Value(sp.auto_produce_n_food), d.GetAllocator());
    j_controls.AddMember("extraMoverFoodCost",     Value(sp.extra_mover_reproductive_cost), d.GetAllocator());

    d.AddMember("controls", j_controls, d.GetAllocator());
}

