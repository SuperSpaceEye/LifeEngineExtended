// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 13.06.22.
//

#include "MainWindow.h"

void MainWindow::write_data(std::ofstream &os) {
    DataSavingFunctions::write_version(os);
    DataSavingFunctions::write_simulation_parameters(os, sp);
    DataSavingFunctions::write_organisms_block_parameters(os, bp);
    DataSavingFunctions::write_data_container_data(os, edc);
    DataSavingFunctions::write_simulation_grid(os, edc);
    DataSavingFunctions::write_organisms(os, edc);
}

//TODO do i need to save spaces?

void MainWindow::recover_state(const SimulationParameters &recovery_sp, const OrganismBlockParameters &recovery_bp,
                               uint32_t recovery_simulation_width, uint32_t recovery_simulation_height) {
    sp = recovery_sp;
    bp = recovery_bp;
    edc.simulation_width  = recovery_simulation_width;
    edc.simulation_height = recovery_simulation_height;
    new_simulation_width  = recovery_simulation_width;
    new_simulation_height = recovery_simulation_height;

    engine.reset_world();
    engine.unpause();
}

void MainWindow::read_data(std::ifstream &is) {
    //If save version is incompatible
    if (!DataSavingFunctions::read_version(is)) {
        display_message("Save version is incompatible with current program version.");
        return;
    }

    engine.partial_clear_world();
    engine.make_walls();

    SimulationParameters recovery_sp = sp;
    OrganismBlockParameters recovery_bp = bp;
    uint32_t recovery_simulation_width = edc.simulation_width;
    uint32_t recovery_simulation_height = edc.simulation_height;

    DataSavingFunctions::read_simulation_parameters(is, sp);
    DataSavingFunctions::read_organisms_block_parameters(is, bp);
    if (read_data_container_data(is)) {
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
    }
    read_simulation_grid(is);
    if (read_organisms(is)) {
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
    }

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

bool MainWindow::read_data_container_data(std::ifstream &is) {
    uint32_t sim_width;
    uint32_t sim_height;

    DataSavingFunctions::read_data_container_data(is, edc, sim_width, sim_height);

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
    ui.cb_fill_window->setChecked(false);
    update_simulation_size_label();
    return false;
}

void MainWindow::read_simulation_grid(std::ifstream &is) {
    just_resize_simulation_grid();

    DataSavingFunctions::read_simulation_grid(is, edc);
}

bool MainWindow::read_organisms(std::ifstream &is) {
    uint32_t num_organisms;
    is.read((char*)&num_organisms, sizeof(uint32_t));

    if (num_organisms > max_loaded_num_organisms) {
        if (!display_dialog_message("The loaded number of organisms is " + std::to_string(num_organisms) +
                                    ". Continue?", false)) {
            return true;
        }
    }

    DataSavingFunctions::read_organisms(is, edc, sp, bp, num_organisms);
    return false;
}

void MainWindow::update_table_values() {
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 4; col++) {
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
            ui.table_organism_block_parameters->item(row, col)->setText(QString::fromStdString(to_str(*value)));
        }
    }
    ui.table_organism_block_parameters->update();
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

    engine.partial_clear_world();
    engine.make_walls();

    json_read_grid_data(document);
    DataSavingFunctions::json_read_simulation_parameters(document, sp);
    DataSavingFunctions::json_read_organisms_data(document, sp, bp, edc);

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

void MainWindow::json_read_grid_data(rapidjson::Document & d) {
    edc.simulation_height = d["num_rows"].GetInt() + 2;
    edc.simulation_width  = d["num_cols"].GetInt() + 2;

    new_simulation_width = edc.simulation_width;
    new_simulation_height = edc.simulation_height;

    fill_window = false;
    ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    update_simulation_size_label();

    just_resize_simulation_grid();

    engine.make_walls();
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

//void MainWindow::write_json_data(const std::string &path) {
//    Document d;
//    d.SetObject();
//
//    auto info = rec.parse_organisms_info();
//
//    d.AddMember("num_rows", Value(edc.simulation_height - 2), d.GetAllocator());
//    d.AddMember("num_cols", Value(edc.simulation_width - 2), d.GetAllocator());
//    d.AddMember("total_mutability", Value(int(info.total_total_mutation_rate*100)), d.GetAllocator());
//    d.AddMember("largest_cell_count", Value(0), d.GetAllocator());
//    d.AddMember("reset_count", Value(0), d.GetAllocator());
//    d.AddMember("total_ticks", Value(edc.total_engine_ticks), d.GetAllocator());
//    d.AddMember("data_update_rate", Value(100), d.GetAllocator());
//
//    json_write_grid(d);
//    json_write_organisms(d);
//    json_write_fossil_record(d);
//    json_write_controls(d);
//
//    StringBuffer buffer;
//    Writer<StringBuffer> writer(buffer);
//    d.Accept(writer);
//
//    std::fstream file;
//    file.open(path, std::ios_base::out);
//    file << buffer.GetString();
//    file.close();
//}
//
//void MainWindow::json_write_grid(rapidjson::Document & d) {
//    Value j_grid(kObjectType);
//    j_grid.AddMember("cols", Value(edc.simulation_width  - 2), d.GetAllocator());
//    j_grid.AddMember("rows", Value(edc.simulation_height - 2), d.GetAllocator());
//
//    Value food(kArrayType);
//    Value walls(kArrayType);
//
//    for (int x = 1; x < edc.simulation_width - 1; x++) {
//        for (int y = 1; y < edc.simulation_height - 1; y++) {
//            if (edc.CPU_simulation_grid[x][y].type != BlockTypes::WallBlock &&
//                edc.CPU_simulation_grid[x][y].type != BlockTypes::FoodBlock) {continue;}
//            Value cell(kObjectType);
//
//            cell.AddMember("c", Value(x-1), d.GetAllocator());
//            cell.AddMember("r", Value(y-1), d.GetAllocator());
//
//            if (edc.CPU_simulation_grid[x][y].type == BlockTypes::FoodBlock) {
//                food.PushBack(cell, d.GetAllocator());
//            } else {
//                walls.PushBack(cell, d.GetAllocator());
//            }
//        }
//    }
//
//    j_grid.AddMember("food", food, d.GetAllocator());
//    j_grid.AddMember("walls", walls, d.GetAllocator());
//
//    d.AddMember("grid", j_grid, d.GetAllocator());
//}
//
//void MainWindow::json_write_organisms(rapidjson::Document & d) {
//    Value j_organisms(kArrayType);
//
//    for (auto & organism_index: edc.organisms) {
//        Value j_organism(kObjectType);
//        write_json_organism(d, organism_index, j_organism);
//        j_organisms.PushBack(j_organism, d.GetAllocator());
//    }
//    d.AddMember("organisms", j_organisms, d.GetAllocator());
//}
//
//void MainWindow::write_json_organism(Document &d, Organism *&organism_index, Value &j_organism) const {
//    Value j_anatomy(kObjectType);
//    Value j_brain(kObjectType);
//    Value cells(kArrayType);
//
//    j_organism.AddMember("c",                Value(organism_index->x-1), d.GetAllocator());
//    j_organism.AddMember("r",                Value(organism_index->y-1), d.GetAllocator());
//    j_organism.AddMember("lifetime",         Value(organism_index->lifetime), d.GetAllocator());
//    j_organism.AddMember("food_collected",   Value(organism_index->food_collected), d.GetAllocator());
//    j_organism.AddMember("living",           Value(true), d.GetAllocator());
//    j_organism.AddMember("direction",        Value(2), d.GetAllocator());
//    j_organism.AddMember("rotation",         Value(static_cast<int>(organism_index->rotation)), d.GetAllocator());
//    j_organism.AddMember("can_rotate",       Value(sp.runtime_rotation_enabled), d.GetAllocator());
//    j_organism.AddMember("move_count",       Value(0), d.GetAllocator());
//    j_organism.AddMember("move_range",       Value(organism_index->move_range), d.GetAllocator());
//    j_organism.AddMember("ignore_brain_for", Value(0), d.GetAllocator());
//    j_organism.AddMember("mutability",       Value(organism_index->anatomy_mutation_rate*100), d.GetAllocator());
//    j_organism.AddMember("damage",           Value(organism_index->damage), d.GetAllocator());
//
//    j_anatomy.AddMember("birth_distance", Value(6), d.GetAllocator());
//    j_anatomy.AddMember("is_producer",    Value(static_cast<bool>(organism_index->anatomy._producer_blocks)), d.GetAllocator());
//    j_anatomy.AddMember("is_mover",       Value(static_cast<bool>(organism_index->anatomy._mover_blocks)), d.GetAllocator());
//    j_anatomy.AddMember("has_eyes",       Value(static_cast<bool>(organism_index->anatomy._eye_blocks)), d.GetAllocator());
//
//    for (auto & block: organism_index->anatomy._organism_blocks) {
//        Value cell(kObjectType);
//        std::string state_name;
//
//        cell.AddMember("loc_col", Value(block.relative_x), d.GetAllocator());
//        cell.AddMember("loc_row", Value(block.relative_y), d.GetAllocator());
//
//        switch (block.type) {
//            case BlockTypes::MouthBlock:    state_name = "mouth";    break;
//            case BlockTypes::ProducerBlock: state_name = "producer"; break;
//            case BlockTypes::MoverBlock:    state_name = "mover";    break;
//            case BlockTypes::KillerBlock:   state_name = "killer";   break;
//            case BlockTypes::ArmorBlock:    state_name = "armor";    break;
//            case BlockTypes::EyeBlock:      state_name = "eye";      break;
//            default: continue;
//        }
//
//        if (block.type == BlockTypes::EyeBlock) {
//            cell.AddMember("direction", Value(static_cast<int>(block.rotation)), d.GetAllocator());
//        }
//
//        Value state(kObjectType);
//        state.AddMember("name", Value(state_name.c_str(), state_name.length(), d.GetAllocator()), d.GetAllocator());
//
//        cell.AddMember("state", state, d.GetAllocator());
//
//        cells.PushBack(cell, d.GetAllocator());
//    }
//    j_anatomy.AddMember("cells", cells, d.GetAllocator());
//
//    j_organism.AddMember("anatomy", j_anatomy, d.GetAllocator());
//
//    auto & table = organism_index->brain.simple_action_table;
//
//    Value decisions(kObjectType);
//
//    decisions.AddMember("empty",    Value(0), d.GetAllocator());
//    decisions.AddMember("food",     Value(static_cast<int>(table.FoodBlock)),     d.GetAllocator());
//    decisions.AddMember("wall",     Value(static_cast<int>(table.WallBlock)),     d.GetAllocator());
//    decisions.AddMember("mouth",    Value(static_cast<int>(table.MouthBlock)),    d.GetAllocator());
//    decisions.AddMember("producer", Value(static_cast<int>(table.ProducerBlock)), d.GetAllocator());
//    decisions.AddMember("mover",    Value(static_cast<int>(table.MoverBlock)),    d.GetAllocator());
//    decisions.AddMember("killer",   Value(static_cast<int>(table.KillerBlock)),   d.GetAllocator());
//    decisions.AddMember("armor",    Value(static_cast<int>(table.ArmorBlock)),    d.GetAllocator());
//    decisions.AddMember("eye",      Value(static_cast<int>(table.EyeBlock)),      d.GetAllocator());
//
//    j_brain.AddMember("decisions", decisions, d.GetAllocator());
//
//    j_organism.AddMember("brain", j_brain, d.GetAllocator());
//
//    j_organism.AddMember("species_name", Value("0000000000"), d.GetAllocator());
//}
//
//void MainWindow::json_write_fossil_record(rapidjson::Document & d) {
//    Value j_fossil_record(kObjectType);
//
//    j_fossil_record.AddMember("min_discard",       Value(10), d.GetAllocator());
//    j_fossil_record.AddMember("record_size_limit", Value(500), d.GetAllocator());
//    j_fossil_record.AddMember("records",           Value(kObjectType), d.GetAllocator());
//    j_fossil_record.AddMember("species",           Value(kObjectType), d.GetAllocator());
//
//    d.AddMember("fossil_record", j_fossil_record, d.GetAllocator());
//}
//
//void MainWindow::json_write_controls(rapidjson::Document & d) const {
//    Value j_controls(kObjectType);
//
//    j_controls.AddMember("lifespanMultiplier", Value(sp.lifespan_multiplier), d.GetAllocator());
//    j_controls.AddMember("foodProdProb", Value(sp.food_production_probability*100), d.GetAllocator());
//
//    Value j_killable_neighbors(kArrayType);
//    Value j_edible_neighbors(kArrayType);
//    Value j_growableNeighbors(kArrayType);
//
//    Value cell(kArrayType);
//    cell.PushBack(Value(0), d.GetAllocator());
//    cell.PushBack(Value(1), d.GetAllocator());
//
//    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//
//    cell = Value(kArrayType);
//    cell.PushBack(Value(0), d.GetAllocator());
//    cell.PushBack(Value(-1), d.GetAllocator());
//
//    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//
//    cell = Value(kArrayType);
//    cell.PushBack(Value(1), d.GetAllocator());
//    cell.PushBack(Value(0), d.GetAllocator());
//
//    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//
//    cell = Value(kArrayType);
//    cell.PushBack(Value(-1), d.GetAllocator());
//    cell.PushBack(Value(0), d.GetAllocator());
//
//    j_killable_neighbors.PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_edible_neighbors  .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//    j_growableNeighbors .PushBack(Value().CopyFrom(cell, d.GetAllocator()), d.GetAllocator());
//
//    j_controls.AddMember("killableNeighbors", j_killable_neighbors, d.GetAllocator());
//    j_controls.AddMember("edibleNeighbors",   j_edible_neighbors,   d.GetAllocator());
//    j_controls.AddMember("growableNeighbors", j_growableNeighbors,  d.GetAllocator());
//
//    j_controls.AddMember("useGlobalMutability",    Value(!sp.use_anatomy_evolved_mutation_rate), d.GetAllocator());
//    j_controls.AddMember("globalMutability",       Value(sp.global_anatomy_mutation_rate*100), d.GetAllocator());
//    j_controls.AddMember("addProb",                Value(sp.add_cell), d.GetAllocator());
//    j_controls.AddMember("changeProb",             Value(sp.change_cell), d.GetAllocator());
//    j_controls.AddMember("removeProb",             Value(sp.remove_cell), d.GetAllocator());
//    j_controls.AddMember("rotationEnabled",        Value(sp.runtime_rotation_enabled), d.GetAllocator());
//    j_controls.AddMember("foodBlocksReproduction", Value(sp.food_blocks_reproduction), d.GetAllocator());
//    j_controls.AddMember("moversCanProduce",       Value(sp.movers_can_produce_food), d.GetAllocator());
//    j_controls.AddMember("instaKill",              Value(sp.on_touch_kill), d.GetAllocator());
//    j_controls.AddMember("lookRange",              Value(sp.look_range), d.GetAllocator());
//    j_controls.AddMember("foodDropProb",           Value(sp.auto_produce_n_food), d.GetAllocator());
//    j_controls.AddMember("extraMoverFoodCost",     Value(sp.extra_mover_reproductive_cost), d.GetAllocator());
//
//    d.AddMember("controls", j_controls, d.GetAllocator());
//}

