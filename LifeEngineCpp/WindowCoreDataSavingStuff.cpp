//
// Created by spaceeye on 13.06.22.
//

#include "WindowCore.h"

void WindowCore::write_data(std::ofstream &os) {
    write_simulation_parameters(os);
    write_organisms_block_parameters(os);
    write_data_container_data(os);
    write_simulation_grid(os);
    write_organisms(os);
}

void WindowCore::write_simulation_parameters(std::ofstream& os) {
    os.write((char*)&sp, sizeof(SimulationParameters));
}

void WindowCore::write_organisms_block_parameters(std::ofstream& os) {
    os.write((char*)&bp, sizeof(BlockParameters));
}

void WindowCore::write_data_container_data(std::ofstream& os) {
    os.write((char*)&dc.total_engine_ticks, sizeof(uint32_t));
    os.write((char*)&dc.simulation_width,   sizeof(uint16_t));
    os.write((char*)&dc.simulation_height,  sizeof(uint16_t));
}
//    void WindowCore::write_color_container(){}
void WindowCore::write_simulation_grid(std::ofstream& os) {
    for (auto & col: dc.CPU_simulation_grid) {
        os.write((char*)&col[0], sizeof(AtomicGridBlock)*col.size());
    }
}
void WindowCore::write_organisms(std::ofstream& os) {
    uint32_t size = dc.organisms.size();
    os.write((char*)&size, sizeof(uint32_t));
    for (auto & organism: dc.organisms) {
        write_organism_data(os, organism);
        write_organism_brain(os, organism->brain.get());
        write_organism_anatomy(os, organism->organism_anatomy.get());
    }
}

void WindowCore::write_organism_data(std::ofstream& os, Organism * organism) {
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

void WindowCore::write_organism_brain(std::ofstream& os, Brain * brain) {
    os.write((char*)brain, sizeof(Brain));
}

//TODO do i need to save spaces?

void WindowCore::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = anatomy->_organism_blocks.size();
    uint32_t producing_space_size                = anatomy->_producing_space.size();
    uint32_t eating_space_size                   = anatomy->_eating_space.size();
    uint32_t killing_space_size                  = anatomy->_killing_space.size();
    uint32_t single_adjacent_space_size          = anatomy->_single_adjacent_space.size();
    uint32_t single_diagonal_adjacent_space_size = anatomy->_single_diagonal_adjacent_space.size();

    os.write((char*)&organism_blocks_size,                sizeof(uint32_t));
    os.write((char*)&producing_space_size,                sizeof(uint32_t));
    os.write((char*)&eating_space_size,                   sizeof(uint32_t));
    os.write((char*)&killing_space_size,                  sizeof(uint32_t));
    os.write((char*)&single_adjacent_space_size,          sizeof(uint32_t));
    os.write((char*)&single_diagonal_adjacent_space_size, sizeof(uint32_t));

    os.write((char*)&anatomy->_mouth_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_producer_blocks, sizeof(int32_t));
    os.write((char*)&anatomy->_mover_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_killer_blocks,   sizeof(int32_t));
    os.write((char*)&anatomy->_armor_blocks,    sizeof(int32_t));
    os.write((char*)&anatomy->_eye_blocks,      sizeof(int32_t));

    os.write((char*)&anatomy->_organism_blocks[0],                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    os.write((char*)&anatomy->_eating_space[0],                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    os.write((char*)&anatomy->_killing_space[0],                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());
    os.write((char*)&anatomy->_single_adjacent_space[0],          sizeof(SerializedArmorSpaceContainer   ) * anatomy->_single_adjacent_space.size());
    os.write((char*)&anatomy->_single_diagonal_adjacent_space[0], sizeof(SerializedAdjacentSpaceContainer) * anatomy->_single_diagonal_adjacent_space.size());

    for (auto & space: anatomy->_producing_space) {
        auto space_size = space.size();
        os.write((char*)&space_size, sizeof(uint32_t));
        os.write((char*)&space[0], sizeof(SerializedAdjacentSpaceContainer) * space_size);
    }
}


void WindowCore::read_data(std::ifstream &is) {
    read_simulation_parameters(is);
    read_organisms_block_parameters(is);
    read_data_container_data(is);
    read_simulation_grid(is);
    read_organisms(is);
}

void WindowCore::read_simulation_parameters(std::ifstream& is) {
    is.read((char*)&sp, sizeof(SimulationParameters));
}

void WindowCore::read_organisms_block_parameters(std::ifstream& is) {
    is.read((char*)&bp, sizeof(BlockParameters));
}

void WindowCore::read_data_container_data(std::ifstream& is) {
    is.read((char*)&dc.total_engine_ticks, sizeof(uint32_t));
    is.read((char*)&dc.simulation_width,   sizeof(uint16_t));
    is.read((char*)&dc.simulation_height,  sizeof(uint16_t));

    new_simulation_width = dc.simulation_width;
    new_simulation_height = dc.simulation_height;
    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
}
//    void WindowCore::read_color_container(){}
void WindowCore::read_simulation_grid(std::ifstream& is) {
    disable_warnings = true;
    resize_simulation_space();
    disable_warnings = false;

    for (auto & col: dc.CPU_simulation_grid) {
        is.read((char*)&col[0], sizeof(AtomicGridBlock)*col.size());
    }

    for (auto & column: dc.CPU_simulation_grid) {
        for (auto &block: column) {
            if (block.type == BlockTypes::WallBlock ||
                block.type == BlockTypes::EmptyBlock ||
                block.type == BlockTypes::FoodBlock) { continue; }
            block = AtomicGridBlock();
        }
    }
}

//TODO save child patterns?
void WindowCore::read_organisms(std::ifstream& is) {
    uint32_t num_organisms;
    is.read((char*)&num_organisms, sizeof(uint32_t));
    dc.organisms.reserve(num_organisms);
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

        dc.organisms.emplace_back(organism);
        SimulationEngineSingleThread::place_organism(&dc, organism);
    }
}

void WindowCore::read_organism_data(std::ifstream& is, OrganismData & data) {
    is.read((char*)&data, sizeof(OrganismData));
}

void WindowCore::read_organism_brain(std::ifstream& is, Brain * brain) {
    is.read((char*)brain, sizeof(Brain));
}

void WindowCore::read_organism_anatomy(std::ifstream& is, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = 0;
    uint32_t producing_space_size                = 0;
    uint32_t eating_space_size                   = 0;
    uint32_t killing_space_size                  = 0;
    uint32_t single_adjacent_space_size          = 0;
    uint32_t single_diagonal_adjacent_space_size = 0;

    is.read((char*)&organism_blocks_size,                sizeof(uint32_t));
    is.read((char*)&producing_space_size,                sizeof(uint32_t));
    is.read((char*)&eating_space_size,                   sizeof(uint32_t));
    is.read((char*)&killing_space_size,                  sizeof(uint32_t));
    is.read((char*)&single_adjacent_space_size,          sizeof(uint32_t));
    is.read((char*)&single_diagonal_adjacent_space_size, sizeof(uint32_t));

    anatomy->_organism_blocks               .resize(organism_blocks_size);
    anatomy->_producing_space               .resize(producing_space_size);
    anatomy->_eating_space                  .resize(eating_space_size);
    anatomy->_killing_space                 .resize(killing_space_size);
    anatomy->_single_adjacent_space         .resize(single_adjacent_space_size);
    anatomy->_single_diagonal_adjacent_space.resize(single_diagonal_adjacent_space_size);

    is.read((char*)&anatomy->_mouth_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_producer_blocks, sizeof(int32_t));
    is.read((char*)&anatomy->_mover_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_killer_blocks,   sizeof(int32_t));
    is.read((char*)&anatomy->_armor_blocks,    sizeof(int32_t));
    is.read((char*)&anatomy->_eye_blocks,      sizeof(int32_t));

    is.read((char*)&anatomy->_organism_blocks[0],                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    is.read((char*)&anatomy->_eating_space[0],                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    is.read((char*)&anatomy->_killing_space[0],                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());
    is.read((char*)&anatomy->_single_adjacent_space[0],          sizeof(SerializedArmorSpaceContainer   ) * anatomy->_single_adjacent_space.size());
    is.read((char*)&anatomy->_single_diagonal_adjacent_space[0], sizeof(SerializedAdjacentSpaceContainer) * anatomy->_single_diagonal_adjacent_space.size());

    for (auto & space: anatomy->_producing_space) {
        uint32_t space_size;
        is.read((char*)&space_size, sizeof(uint32_t));
        space.resize(space_size);
        is.read((char*)&space[0], sizeof(SerializedAdjacentSpaceContainer) * space_size);
    }
}

void WindowCore::update_table_values() {
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 3; col++) {
            BlockParameters *type;
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
void WindowCore::read_json_data(std::string path) {
    pt::ptree root;
    pt::read_json(path, root);

    json_read_grid_data(root);

    json_read_simulation_parameters(root);

    json_read_organism_data(root);
}

void WindowCore::json_read_grid_data(boost::property_tree::ptree &root) {
    dc.simulation_height  = root.get<int>("num_rows") + 2;
    dc.simulation_width   = root.get<int>("num_cols") + 2;

    new_simulation_width = dc.simulation_width;
    new_simulation_height = dc.simulation_height;

    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    resize_simulation_space();
    make_walls();
    disable_warnings = false;

    dc.total_engine_ticks = root.get<int>("total_ticks");

    for (auto & pair: root.get_child("grid.food")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        dc.CPU_simulation_grid[x][y].type = FoodBlock;
    }

    for (auto & pair: root.get_child("grid.walls")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        dc.CPU_simulation_grid[x][y].type = WallBlock;
    }
}

void WindowCore::json_read_simulation_parameters(const boost::property_tree::ptree &root) {
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
//    sp.auto_produce_n_food = root.get<int>("controls.foodDropProb");
//     = root.get<int>("controls.extraMoverFoodCost");

}

void WindowCore::json_read_organism_data(boost::property_tree::ptree &root) {
    for (auto & organism: root.get_child("organisms")) {
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        int y              = organism.second.get<int>("r")+1;
        int x              = organism.second.get<int>("c")+1;
        int lifetime       = organism.second.get<int>("lifetime");
        int food_collected = organism.second.get<int>("food_collected");
        int rotation       = organism.second.get<int>("rotation");
        int move_range     = organism.second.get<int>("move_range");
        float mutability   = float(organism.second.get<int>("mutability"))/100;
        int damage         = organism.second.get<int>("damage");
        bool is_mover      = organism.second.get<bool>("anatomy.is_mover");
        bool has_eyes      = organism.second.get<bool>("anatomy.has_eyes");

        for (auto & cell: organism.second.get_child("anatomy.cells")) {
            int l_y = cell.second.get<int>("loc_row");
            int l_x = cell.second.get<int>("loc_col");
            auto state = cell.second.get<std::string>("state.name");

            Rotation rotation = Rotation::UP;
            BlockTypes type = ProducerBlock;

            if        (state == "producer") {
                type = ProducerBlock;
            } else if (state == "mouth") {
                type = MouthBlock;
            } else if (state == "killer") {
                type = KillerBlock;
            } else if (state == "mover") {
                type = MoverBlock;
            } else if (state == "eye") {
                type = EyeBlock;
                rotation = static_cast<Rotation>(cell.second.get<int>("direction"));
            } else if (state == "armor") {
                type = ArmorBlock;
            }
            anatomy->set_block(type, rotation, l_x, l_y);
        }
        if (is_mover && has_eyes) {
            brain->simple_action_table.FoodBlock     = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.food"));
            brain->simple_action_table.WallBlock     = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.wall"));
            brain->simple_action_table.MouthBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.mouth"));
            brain->simple_action_table.ProducerBlock = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.producer"));
            brain->simple_action_table.MoverBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.mover"));
            brain->simple_action_table.KillerBlock   = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.killer"));
            brain->simple_action_table.ArmorBlock    = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.armor"));
            brain->simple_action_table.EyeBlock      = static_cast<SimpleDecision>(organism.second.get<int>("brain.decisions.eye"));
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
        dc.organisms.emplace_back(new_organism);
        SimulationEngineSingleThread::place_organism(&dc, new_organism);
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

void WindowCore::write_json_data(std::string path) {
    pt::ptree root, grid, organisms, fossil_record, controls, killable_neighbors, edible_neighbors, growableNeighbors, cell, value, anatomy, cells, j_organism, food, walls, brain;

    auto info = calculate_organisms_info();

    root.put("num_rows", dc.simulation_height);
    root.put("num_cols", dc.simulation_width);
    root.put("total_mutability", static_cast<int>(info.total_total_mutation_rate*100));
    root.put("largest_cell_count", 0);
    root.put("reset_count", auto_reset_num);
    root.put("total_ticks", dc.total_engine_ticks);
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

void WindowCore::json_write_grid(boost::property_tree::ptree &grid, boost::property_tree::ptree &cell,
                                 boost::property_tree::ptree &food, boost::property_tree::ptree &walls) {
    grid.put("cols", dc.simulation_width);
    grid.put("rows", dc.simulation_height);

    for (int x = 0; x < dc.simulation_width; x++) {
        for (int y = 0; y < dc.simulation_height; y++) {
            if (dc.CPU_simulation_grid[x][y].type != WallBlock &&
                dc.CPU_simulation_grid[x][y].type != FoodBlock) {continue;}
            cell = pt::ptree{};

            cell.put("c", x);
            cell.put("r", y);
            if (dc.CPU_simulation_grid[x][y].type == FoodBlock) {
                food.push_back(std::make_pair("", cell));
            } else {
                walls.push_back(std::make_pair("", cell));
            }
        }
    }

    grid.put_child("food", food);
    grid.put_child("walls", walls);
}

void WindowCore::json_write_organisms(boost::property_tree::ptree &organisms, boost::property_tree::ptree &cell,
                                      boost::property_tree::ptree &anatomy, boost::property_tree::ptree &cells,
                                      boost::property_tree::ptree &j_organism, boost::property_tree::ptree &brain) {
    for (auto & organism: dc.organisms) {
        j_organism = pt::ptree{};
        anatomy = pt::ptree{};
        cells = pt::ptree{};
        brain = pt::ptree{};

        j_organism.put("c", organism->x);
        j_organism.put("r", organism->y);
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
        anatomy.put("is_producer", static_cast<bool>(organism->organism_anatomy->_producer_blocks));
        anatomy.put("is_mover", static_cast<bool>(organism->organism_anatomy->_mover_blocks));
        anatomy.put("has_eyes", static_cast<bool>(organism->organism_anatomy->_eye_blocks));

        for (auto & block: organism->organism_anatomy->_organism_blocks) {
            cell = pt::ptree{};
            std::string state_name;

            cell.put("loc_col", block.relative_x);
            cell.put("loc_row", block.relative_y);

            switch (block.type) {
                case MouthBlock: state_name    = "mouth";    break;
                case ProducerBlock: state_name = "producer"; break;
                case MoverBlock: state_name    = "mover";    break;
                case KillerBlock: state_name   = "killer";   break;
                case ArmorBlock: state_name    = "armor";    break;
                case EyeBlock: state_name      = "eye";      break;
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

        brain.put("decisions.empty", 0);
        brain.put("decisions.food",     static_cast<int>(organism->brain->simple_action_table.FoodBlock));
        brain.put("decisions.wall",     static_cast<int>(organism->brain->simple_action_table.WallBlock));
        brain.put("decisions.mouth",    static_cast<int>(organism->brain->simple_action_table.MouthBlock));
        brain.put("decisions.producer", static_cast<int>(organism->brain->simple_action_table.ProducerBlock));
        brain.put("decisions.mover",    static_cast<int>(organism->brain->simple_action_table.MoverBlock));
        brain.put("decisions.killer",   static_cast<int>(organism->brain->simple_action_table.KillerBlock));
        brain.put("decisions.armor",    static_cast<int>(organism->brain->simple_action_table.ArmorBlock));
        brain.put("decisions.eye",      static_cast<int>(organism->brain->simple_action_table.EyeBlock));

        j_organism.put_child("brain", brain);

        j_organism.put("species_name", "0000000000", my_id_translator<std::string>());

        organisms.push_back(std::make_pair("", j_organism));
    }
}

void WindowCore::json_write_fossil_record(boost::property_tree::ptree &fossil_record) const {
    fossil_record.put("min_discard", 10);
    fossil_record.put("record_size_limit", 500);
    fossil_record.put("records", "{}");
    fossil_record.put("species", "{}");
}

void
WindowCore::json_write_controls(boost::property_tree::ptree &controls, boost::property_tree::ptree &killable_neighbors,
                                boost::property_tree::ptree &edible_neighbors,
                                boost::property_tree::ptree &growableNeighbors, boost::property_tree::ptree &cell,
                                boost::property_tree::ptree &value) const {
    controls.put("lifespanMultiplier", static_cast<int>(sp.lifespan_multiplier));
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