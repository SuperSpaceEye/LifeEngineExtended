//
// Created by spaceeye on 31.07.22.
//

#include "DataSavingFunctions.h"

//TODO increment every time saving logic changes
const uint32_t SAVE_VERSION = 6;

void DataSavingFunctions::write_version(std::ofstream &os) {
    os.write((char*)&SAVE_VERSION, sizeof(uint32_t));
}

void DataSavingFunctions::write_simulation_parameters(std::ofstream & os, SimulationParameters &sp) {
    os.write((char*)&sp, sizeof(SimulationParameters));
}

void DataSavingFunctions::write_organisms_block_parameters(std::ofstream & os, OrganismBlockParameters &bp) {
    os.write((char*)&bp, sizeof(OrganismBlockParameters));
}

void DataSavingFunctions::write_data_container_data(std::ofstream & os, EngineDataContainer &edc) {
    os.write((char*)&edc.total_engine_ticks, sizeof(uint32_t));
    os.write((char*)&edc.simulation_width,   sizeof(uint32_t));
    os.write((char*)&edc.simulation_height,  sizeof(uint32_t));
}

void DataSavingFunctions::write_simulation_grid(std::ofstream & os, EngineDataContainer &edc) {
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

void DataSavingFunctions::write_organism(std::ofstream &os, Organism *organism) {
    write_organism_brain(os,   &organism->brain);
    write_organism_anatomy(os, &organism->anatomy);
    write_organism_data(os,    organism);
}

void DataSavingFunctions::write_organisms(std::ofstream & os, EngineDataContainer &edc) {
    uint32_t size = edc.stc.num_alive_organisms;
    os.write((char*)&size, sizeof(uint32_t));
//    for (auto & organism_index: edc.stc.organisms) {
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        auto & organism = edc.stc.organisms[i];
        if (organism.is_dead) {continue;}
        write_organism(os, &organism);
    }
}

void DataSavingFunctions::write_organism_data(std::ofstream & os, Organism * organism) {
    os.write((char*)static_cast<OrganismData*>(organism), sizeof(OrganismData));
}

void DataSavingFunctions::write_organism_brain(std::ofstream & os, Brain * brain) {
    os.write((char*)brain, sizeof(Brain));
}

void DataSavingFunctions::write_organism_anatomy(std::ofstream & os, Anatomy * anatomy) {
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



bool DataSavingFunctions::read_version(std::ifstream &is) {
    int save_version;
    is.read((char*)&save_version, sizeof(int));
    return save_version == SAVE_VERSION;
}

void DataSavingFunctions::read_simulation_parameters(std::ifstream& is, SimulationParameters &sp) {
    is.read((char*)&sp, sizeof(SimulationParameters));
}

void DataSavingFunctions::read_organisms_block_parameters(std::ifstream& is, OrganismBlockParameters &bp) {
    is.read((char*)&bp, sizeof(OrganismBlockParameters));
}

void DataSavingFunctions::read_data_container_data(std::ifstream& is, EngineDataContainer &edc, uint32_t &sim_width, uint32_t &sim_height) {
    is.read((char*)&edc.loaded_engine_ticks, sizeof(uint32_t));
    is.read((char*)&sim_width,    sizeof(uint32_t));
    is.read((char*)&sim_height,   sizeof(uint32_t));
}

void DataSavingFunctions::read_simulation_grid(std::ifstream& is, EngineDataContainer &edc) {
    std::vector<WorldBlocks> blocks{};
    std::size_t size;

    is.read((char*)&size, sizeof(std::size_t));
    blocks.resize(size);

    is.read((char*)&blocks[0], sizeof(WorldBlocks)*size);

    for (auto & block: blocks) {
        edc.CPU_simulation_grid[block.x][block.y].type = block.type;
    }
}

void DataSavingFunctions::read_organism(std::ifstream &is, SimulationParameters &sp, OrganismBlockParameters &bp,
                                        Organism *organism) {
    auto brain = Brain();
    auto anatomy = Anatomy();

    read_organism_brain(is, &brain);
    read_organism_anatomy(is, &anatomy);

    *organism = Organism(0,
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
}

//TODO save child patterns?
bool DataSavingFunctions::read_organisms(std::ifstream& is, EngineDataContainer &edc, SimulationParameters &sp, OrganismBlockParameters &bp, uint32_t num_organisms) {
    edc.stc.organisms.reserve(num_organisms);
    for (int i = 0; i < num_organisms; i++) {
        auto * organism = OrganismsController::get_new_main_organism(edc);
        auto array_place = organism->vector_index;
        read_organism(is, sp, bp, organism);
        organism->vector_index = array_place;
        SimulationEngineSingleThread::place_organism(&edc, organism);
    }

    return false;
}

void DataSavingFunctions::read_organism_data(std::ifstream& is, OrganismData & data) {
    is.read((char*)&data, sizeof(OrganismData));
}

void DataSavingFunctions::read_organism_brain(std::ifstream& is, Brain * brain) {
    is.read((char*)brain, sizeof(Brain));
}

void DataSavingFunctions::read_organism_anatomy(std::ifstream& is, Anatomy * anatomy) {
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

using rapidjson::Value, rapidjson::Document, rapidjson::StringBuffer, rapidjson::Writer, rapidjson::kObjectType, rapidjson::kArrayType;

void DataSavingFunctions::write_json_data(const std::string &path, EngineDataContainer &edc, SimulationParameters &sp, double total_total_mutation_rate) {
    Document d;
    d.SetObject();

    d.AddMember("num_rows", Value(edc.simulation_height - 2), d.GetAllocator());
    d.AddMember("num_cols", Value(edc.simulation_width - 2), d.GetAllocator());
    d.AddMember("total_mutability", Value(int(total_total_mutation_rate*100)), d.GetAllocator());
    d.AddMember("largest_cell_count", Value(0), d.GetAllocator());
    d.AddMember("reset_count", Value(0), d.GetAllocator());
    d.AddMember("total_ticks", Value(edc.total_engine_ticks), d.GetAllocator());
    d.AddMember("data_update_rate", Value(100), d.GetAllocator());

    json_write_grid(d, edc);
    json_write_organisms(d, edc, sp);
    json_write_fossil_record(d);
    json_write_controls(d, sp);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    std::fstream file;
    file.open(path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}

void DataSavingFunctions::json_write_grid(Document & d, EngineDataContainer &edc) {
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

void DataSavingFunctions::json_write_organisms(Document & d, EngineDataContainer &edc, SimulationParameters &sp) {
    Value j_organisms(kArrayType);

    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        auto * organism = &edc.stc.organisms[i];
        if (organism->is_dead) { continue;}

        Value j_organism(kObjectType);
        write_json_organism(d, organism, j_organism, sp);
        j_organisms.PushBack(j_organism, d.GetAllocator());
    }
    d.AddMember("organisms", j_organisms, d.GetAllocator());
}

void DataSavingFunctions::write_json_organism(Document &d, Organism * organism, Value &j_organism, SimulationParameters &sp) {
    Value j_anatomy(kObjectType);
    Value j_brain(kObjectType);
    Value cells(kArrayType);

    j_organism.AddMember("c",                Value(organism->x-1), d.GetAllocator());
    j_organism.AddMember("r",                Value(organism->y-1), d.GetAllocator());
    j_organism.AddMember("lifetime",         Value(organism->lifetime), d.GetAllocator());
    j_organism.AddMember("food_collected",   Value(organism->food_collected), d.GetAllocator());
    j_organism.AddMember("living",           Value(true), d.GetAllocator());
    j_organism.AddMember("direction",        Value(2), d.GetAllocator());
    //The rotation in the original version goes Up, right, down, left. while in mine it is up, left, down, right.
    auto rotation = static_cast<int>(organism->rotation);
    if (rotation == 1) {rotation = 3;}
    else if (rotation == 3) {rotation = 1;}
    j_organism.AddMember("rotation",         Value(rotation), d.GetAllocator());
    j_organism.AddMember("can_rotate",       Value(sp.runtime_rotation_enabled), d.GetAllocator());
    j_organism.AddMember("move_count",       Value(0), d.GetAllocator());
    j_organism.AddMember("move_range",       Value(organism->move_range), d.GetAllocator());
    j_organism.AddMember("ignore_brain_for", Value(0), d.GetAllocator());
    j_organism.AddMember("mutability",       Value(organism->anatomy_mutation_rate*100), d.GetAllocator());
    j_organism.AddMember("damage",           Value(organism->damage), d.GetAllocator());

    j_anatomy.AddMember("birth_distance", Value(6), d.GetAllocator());
    j_anatomy.AddMember("is_producer",    Value(static_cast<bool>(organism->anatomy._producer_blocks)), d.GetAllocator());
    j_anatomy.AddMember("is_mover",       Value(static_cast<bool>(organism->anatomy._mover_blocks)), d.GetAllocator());
    j_anatomy.AddMember("has_eyes",       Value(static_cast<bool>(organism->anatomy._eye_blocks)), d.GetAllocator());

    for (auto & block: organism->anatomy._organism_blocks) {
        Value cell(kObjectType);
        std::string state_name;

        cell.AddMember("loc_col", Value(block.relative_x), d.GetAllocator());
        cell.AddMember("loc_row", Value(block.relative_y), d.GetAllocator());

        switch (block.type) {
            case BlockTypes::MouthBlock:    state_name = "mouth";    break;
            case BlockTypes::ProducerBlock: state_name = "producer"; break;
            case BlockTypes::MoverBlock:    state_name = "mover";    break;
            case BlockTypes::KillerBlock:   state_name = "killer";   break;
            case BlockTypes::ArmorBlock:    state_name = "armor";    break;
            case BlockTypes::EyeBlock:      state_name = "eye";      break;
            default: continue;
        }

        if (block.type == BlockTypes::EyeBlock) {
            auto rotation = block.rotation;
            if (rotation == Rotation::RIGHT)     {rotation = Rotation::LEFT;}
            else if (rotation == Rotation::LEFT) {rotation = Rotation::RIGHT;}
            cell.AddMember("direction", Value(static_cast<int>(rotation)), d.GetAllocator());
        }

        Value state(kObjectType);
        state.AddMember("name", Value(state_name.c_str(), state_name.length(), d.GetAllocator()), d.GetAllocator());

        cell.AddMember("state", state, d.GetAllocator());

        cells.PushBack(cell, d.GetAllocator());
    }
    j_anatomy.AddMember("cells", cells, d.GetAllocator());

    j_organism.AddMember("anatomy", j_anatomy, d.GetAllocator());

    auto & table = organism->brain.simple_action_table;

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

void DataSavingFunctions::json_write_fossil_record(rapidjson::Document & d) {
    Value j_fossil_record(kObjectType);

    j_fossil_record.AddMember("min_discard",       Value(10), d.GetAllocator());
    j_fossil_record.AddMember("record_size_limit", Value(500), d.GetAllocator());
    j_fossil_record.AddMember("records",           Value(kObjectType), d.GetAllocator());
    j_fossil_record.AddMember("species",           Value(kObjectType), d.GetAllocator());

    d.AddMember("fossil_record", j_fossil_record, d.GetAllocator());
}

void DataSavingFunctions::json_write_controls(rapidjson::Document & d, SimulationParameters &sp) {
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

void DataSavingFunctions::json_read_simulation_parameters(rapidjson::Document * d_, SimulationParameters * sp_) {
    auto &sp = *sp_;
    auto &d = *d_;

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

void DataSavingFunctions::json_read_organism(rapidjson::GenericValue<rapidjson::UTF8<>> &organism,
                                             SimulationParameters &sp,
                                             OrganismBlockParameters &bp, Organism *new_organism) {
    auto brain = Brain();
    auto anatomy = Anatomy();

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

        if (_rotation == Rotation::RIGHT)     {_rotation = Rotation::LEFT;}
        else if (_rotation == Rotation::LEFT) {_rotation = Rotation::RIGHT;}

        block_data.emplace_back(type, _rotation, l_x, l_y);
    }
    anatomy.set_many_blocks(block_data);

    if (is_mover && has_eyes) {
        auto & table = brain.simple_action_table;
        table.FoodBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["food"]    .GetInt());
        table.WallBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["wall"]    .GetInt());
        table.MouthBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mouth"]   .GetInt());
        table.ProducerBlock = static_cast<SimpleDecision>(organism["brain"]["decisions"]["producer"].GetInt());
        table.MoverBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mover"]   .GetInt());
        table.KillerBlock   = static_cast<SimpleDecision>(organism["brain"]["decisions"]["killer"]  .GetInt());
        table.ArmorBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["armor"]   .GetInt());
        table.EyeBlock      = static_cast<SimpleDecision>(organism["brain"]["decisions"]["eye"]     .GetInt());
    }

    if (rotation == 1) {rotation = 3;}
    else if (rotation == 3) {rotation = 1;}

    *new_organism = Organism(x,
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
}

void DataSavingFunctions::json_read_organisms_data(Document *d_, SimulationParameters *sp_, OrganismBlockParameters *bp_, EngineDataContainer *edc_) {
    auto &d = *d_;
    auto &sp = *sp_;
    auto &bp = *bp_;
    auto &edc = *edc_;

    for (auto & organism: d["organisms"].GetArray()) {
        auto new_organism = OrganismsController::get_new_main_organism(edc);
        auto array_place = new_organism->vector_index;
        json_read_organism(organism, sp, bp, new_organism);
        new_organism->vector_index = array_place;

        if (new_organism->anatomy._mover_blocks > 0 && new_organism->anatomy._eye_blocks > 0) {new_organism->brain.brain_type = BrainTypes::SimpleBrain;}

        SimulationEngineSingleThread::place_organism(&edc, new_organism);
    }

    auto gen = lehmer64(42);

    for (auto & organism: edc.stc.organisms) {
        edc.stc.organisms_observations.clear();
        SimulationEngineSingleThread::reserve_observations(edc.stc.organisms_observations, edc.stc.organisms, &edc);
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {SimulationEngineSingleThread::get_observations(&edc, &sp, &organism, edc.stc.organisms_observations);}}

        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {organism.think_decision(edc.stc.organisms_observations[i],
                                                                                                                                                          &gen);}}
    }
}


const uint32_t STATE_SAVE_VERSION = 1;

void DataSavingFunctions::write_json_version(rapidjson::Document & d) {
    d.AddMember("save_version", Value(STATE_SAVE_VERSION), d.GetAllocator());
}

bool DataSavingFunctions::read_json_version(rapidjson::Document & d) {
    if (!d.HasMember("save_version")) {return false;}
    auto version = d["save_version"].GetUint();
    return version == STATE_SAVE_VERSION;
}

void DataSavingFunctions::write_json_extended_simulation_parameters(rapidjson::Document &d, SimulationParameters &sp) {
    d.AddMember("food_production_probability",     Value(sp.food_production_probability), d.GetAllocator());
    d.AddMember("produce_food_every_n_life_ticks", Value(sp.produce_food_every_n_life_ticks), d.GetAllocator());
    d.AddMember("lifespan_multiplier",             Value(sp.lifespan_multiplier), d.GetAllocator());
    d.AddMember("look_range",                      Value(sp.look_range), d.GetAllocator());
    d.AddMember("auto_produce_food_every_n_ticks", Value(sp.auto_produce_food_every_n_ticks), d.GetAllocator());
    d.AddMember("auto_produce_n_food",             Value(sp.auto_produce_n_food), d.GetAllocator());
    d.AddMember("extra_reproduction_cost",         Value(sp.extra_reproduction_cost), d.GetAllocator());
    d.AddMember("extra_mover_reproductive_cost",   Value(sp.extra_mover_reproductive_cost), d.GetAllocator());
    d.AddMember("global_anatomy_mutation_rate",    Value(sp.global_anatomy_mutation_rate), d.GetAllocator());
    d.AddMember("global_brain_mutation_rate",      Value(sp.global_brain_mutation_rate), d.GetAllocator());
    d.AddMember("killer_damage_amount",            Value(sp.killer_damage_amount), d.GetAllocator());

    d.AddMember("min_reproducing_distance", Value(sp.min_reproducing_distance), d.GetAllocator());
    d.AddMember("max_reproducing_distance", Value(sp.max_reproducing_distance), d.GetAllocator());

    d.AddMember("anatomy_mutations_rate_mutation_step", Value(sp.anatomy_mutations_rate_mutation_step), d.GetAllocator());
    d.AddMember("anatomy_min_possible_mutation_rate",   Value(sp.anatomy_min_possible_mutation_rate), d.GetAllocator());
    d.AddMember("anatomy_mutation_rate_delimiter",      Value(sp.anatomy_mutation_rate_delimiter), d.GetAllocator());

    d.AddMember("brain_mutation_rate_mutation_step", Value(sp.brain_mutation_rate_mutation_step), d.GetAllocator());
    d.AddMember("brain_min_possible_mutation_rate",  Value(sp.brain_min_possible_mutation_rate), d.GetAllocator());
    d.AddMember("brain_mutation_rate_delimiter",     Value(sp.brain_mutation_rate_delimiter), d.GetAllocator());

    d.AddMember("min_move_range",       Value(sp.min_move_range), d.GetAllocator());
    d.AddMember("max_move_range",       Value(sp.max_move_range), d.GetAllocator());
    d.AddMember("move_range_delimiter", Value(sp.move_range_delimiter), d.GetAllocator());
    d.AddMember("set_fixed_move_range", Value(sp.set_fixed_move_range), d.GetAllocator());
    d.AddMember("min_organism_size",    Value(sp.min_organism_size), d.GetAllocator());

    d.AddMember("add_cell",    Value(sp.add_cell), d.GetAllocator());
    d.AddMember("change_cell", Value(sp.change_cell), d.GetAllocator());
    d.AddMember("remove_cell", Value(sp.remove_cell), d.GetAllocator());

    d.AddMember("perlin_octaves",     Value(sp.perlin_octaves), d.GetAllocator());
    d.AddMember("perlin_persistence", Value(sp.perlin_persistence), d.GetAllocator());
    d.AddMember("perlin_upper_bound", Value(sp.perlin_upper_bound), d.GetAllocator());
    d.AddMember("perlin_lower_bound", Value(sp.perlin_lower_bound), d.GetAllocator());
    d.AddMember("perlin_x_modifier",  Value(sp.perlin_x_modifier), d.GetAllocator());
    d.AddMember("perlin_y_modifier",  Value(sp.perlin_y_modifier), d.GetAllocator());

    d.AddMember("reproduction_rotation_enabled",     Value(sp.reproduction_rotation_enabled), d.GetAllocator());
    d.AddMember("on_touch_kill",                     Value(sp.on_touch_kill), d.GetAllocator());
    d.AddMember("use_anatomy_evolved_mutation_rate", Value(sp.use_anatomy_evolved_mutation_rate), d.GetAllocator());
    d.AddMember("use_brain_evolved_mutation_rate",   Value(sp.use_brain_evolved_mutation_rate), d.GetAllocator());
    d.AddMember("movers_can_produce_food",           Value(sp.movers_can_produce_food), d.GetAllocator());
    d.AddMember("food_blocks_reproduction",          Value(sp.food_blocks_reproduction), d.GetAllocator());

    d.AddMember("reset_on_total_extinction",        Value(sp.reset_on_total_extinction), d.GetAllocator());
    d.AddMember("pause_on_total_extinction",        Value(sp.pause_on_total_extinction), d.GetAllocator());
    d.AddMember("clear_walls_on_reset",             Value(sp.clear_walls_on_reset), d.GetAllocator());
    d.AddMember("generate_random_walls_on_reset",   Value(sp.generate_random_walls_on_reset), d.GetAllocator());
    d.AddMember("reproduction_distance_fixed",      Value(sp.reproduction_distance_fixed), d.GetAllocator());
    d.AddMember("runtime_rotation_enabled",         Value(sp.runtime_rotation_enabled), d.GetAllocator());
    d.AddMember("organism_self_blocks_block_sight", Value(sp.organism_self_blocks_block_sight), d.GetAllocator());
    d.AddMember("failed_reproduction_eats_food",    Value(sp.failed_reproduction_eats_food), d.GetAllocator());
    d.AddMember("rotate_every_move_tick",           Value(sp.rotate_every_move_tick), d.GetAllocator());
    d.AddMember("multiply_food_production_prob",    Value(sp.multiply_food_production_prob), d.GetAllocator());
    d.AddMember("simplified_food_production",       Value(sp.simplified_food_production), d.GetAllocator());
    d.AddMember("stop_when_one_food_generated",     Value(sp.stop_when_one_food_generated), d.GetAllocator());
    d.AddMember("eat_then_produce",                 Value(sp.eat_then_produce), d.GetAllocator());
    d.AddMember("check_if_path_is_clear",           Value(sp.check_if_path_is_clear), d.GetAllocator());
    d.AddMember("food_blocks_movement",             Value(sp.food_blocks_movement), d.GetAllocator());
    d.AddMember("use_new_child_pos_calculator",     Value(sp.use_new_child_pos_calculator), d.GetAllocator());
    d.AddMember("no_random_decisions",              Value(sp.no_random_decisions), d.GetAllocator());
}

void DataSavingFunctions::read_json_extended_simulation_parameters(rapidjson::Document &d, SimulationParameters &sp) {
    sp.food_production_probability     = d["food_production_probability"].GetFloat();
    sp.produce_food_every_n_life_ticks = d["produce_food_every_n_life_ticks"].GetInt();
    sp.lifespan_multiplier             = d["lifespan_multiplier"].GetFloat();
    sp.look_range                      = d["look_range"].GetInt();
    sp.auto_produce_food_every_n_ticks = d["auto_produce_food_every_n_ticks"].GetInt();
    sp.auto_produce_n_food             = d["auto_produce_n_food"].GetInt();
    sp.extra_reproduction_cost         = d["extra_reproduction_cost"].GetFloat();
    sp.extra_mover_reproductive_cost   = d["extra_mover_reproductive_cost"].GetFloat();
    sp.global_anatomy_mutation_rate    = d["global_anatomy_mutation_rate"].GetFloat();
    sp.global_brain_mutation_rate      = d["global_brain_mutation_rate"].GetFloat();
    sp.killer_damage_amount            = d["killer_damage_amount"].GetFloat();

    sp.min_reproducing_distance    = d["min_reproducing_distance"].GetInt();
    sp.max_reproducing_distance    = d["max_reproducing_distance"].GetInt();

    sp.anatomy_mutations_rate_mutation_step = d["anatomy_mutations_rate_mutation_step"].GetFloat();
    sp.anatomy_min_possible_mutation_rate   = d["anatomy_min_possible_mutation_rate"].GetFloat();
    sp.anatomy_mutation_rate_delimiter      = d["anatomy_mutation_rate_delimiter"].GetFloat();

    sp.brain_mutation_rate_mutation_step = d["brain_mutation_rate_mutation_step"].GetFloat();
    sp.brain_min_possible_mutation_rate  = d["brain_min_possible_mutation_rate"].GetFloat();
    sp.brain_mutation_rate_delimiter     = d["brain_mutation_rate_delimiter"].GetFloat();

    sp.min_move_range       = d["min_move_range"].GetInt();
    sp.max_move_range       = d["max_move_range"].GetInt();
    sp.move_range_delimiter = d["move_range_delimiter"].GetFloat();
    sp.set_fixed_move_range = d["set_fixed_move_range"].GetBool();
    sp.min_organism_size    = d["min_organism_size"].GetInt();

    sp.add_cell    = d["add_cell"].GetInt();
    sp.change_cell = d["change_cell"].GetInt();
    sp.remove_cell = d["remove_cell"].GetInt();

    sp.perlin_octaves     = d["perlin_octaves"].GetInt();
    sp.perlin_persistence = d["perlin_persistence"].GetFloat();
    sp.perlin_upper_bound = d["perlin_upper_bound"].GetFloat();
    sp.perlin_lower_bound = d["perlin_lower_bound"].GetFloat();
    sp.perlin_x_modifier  = d["perlin_x_modifier"].GetFloat();
    sp.perlin_y_modifier  = d["perlin_y_modifier"].GetFloat();

    sp.reproduction_rotation_enabled     = d["reproduction_rotation_enabled"].GetBool();
    sp.on_touch_kill                     = d["on_touch_kill"].GetBool();
    sp.use_anatomy_evolved_mutation_rate = d["use_anatomy_evolved_mutation_rate"].GetBool();
    sp.use_brain_evolved_mutation_rate   = d["use_brain_evolved_mutation_rate"].GetBool();
    sp.movers_can_produce_food           = d["movers_can_produce_food"].GetBool();
    sp.food_blocks_reproduction          = d["food_blocks_reproduction"].GetBool();

    sp.reset_on_total_extinction        = d["reset_on_total_extinction"].GetBool();
    sp.pause_on_total_extinction        = d["pause_on_total_extinction"].GetBool();
    sp.clear_walls_on_reset             = d["clear_walls_on_reset"].GetBool();
    sp.generate_random_walls_on_reset   = d["generate_random_walls_on_reset"].GetBool();
    sp.reproduction_distance_fixed      = d["reproduction_distance_fixed"].GetBool();
    sp.runtime_rotation_enabled         = d["runtime_rotation_enabled"].GetBool();
    sp.organism_self_blocks_block_sight = d["organism_self_blocks_block_sight"].GetBool();
    sp.failed_reproduction_eats_food    = d["failed_reproduction_eats_food"].GetBool();
    sp.rotate_every_move_tick           = d["rotate_every_move_tick"].GetBool();
    sp.multiply_food_production_prob    = d["multiply_food_production_prob"].GetBool();
    sp.simplified_food_production       = d["simplified_food_production"].GetBool();
    sp.stop_when_one_food_generated     = d["stop_when_one_food_generated"].GetBool();
    sp.eat_then_produce                 = d["eat_then_produce"].GetBool();
    sp.check_if_path_is_clear           = d["check_if_path_is_clear"].GetBool();
    sp.food_blocks_movement             = d["food_blocks_movement"].GetBool();
    sp.use_new_child_pos_calculator     = d["use_new_child_pos_calculator"].GetBool();
    sp.no_random_decisions              = d["no_random_decisions"].GetBool();
}


void DataSavingFunctions::write_json_program_settings(rapidjson::Document &d, DataSavingFunctions::ProgramState &state) {
    d.AddMember("scaling_coefficient",                Value(state.scaling_coefficient), d.GetAllocator());
    d.AddMember("keyboard_movement_amount",           Value(state.keyboard_movement_amount), d.GetAllocator());
    d.AddMember("SHIFT_keyboard_movement_multiplier", Value(state.SHIFT_keyboard_movement_multiplier), d.GetAllocator());

    d.AddMember("font_size",                        Value(state.font_size), d.GetAllocator());
    d.AddMember("float_precision",                  Value(state.float_precision), d.GetAllocator());
    d.AddMember("brush_size",                       Value(state.brush_size), d.GetAllocator());
    d.AddMember("update_info_every_n_milliseconds", Value(state.update_info_every_n_milliseconds), d.GetAllocator());

    d.AddMember("use_cuda",                          Value(state.use_cuda), d.GetAllocator());
    d.AddMember("wait_for_engine_to_stop_to_render", Value(state.wait_for_engine_to_stop_to_render), d.GetAllocator());
    d.AddMember("disable_warnings",                  Value(state.disable_warnings), d.GetAllocator());
    d.AddMember("really_stop_render",                Value(state.really_stop_render), d.GetAllocator());
    d.AddMember("save_simulation_settings",          Value(state.save_simulation_settings), d.GetAllocator());
    d.AddMember("use_point_size",                    Value(state.use_point_size), d.GetAllocator());
}

void DataSavingFunctions::read_json_program_settings(rapidjson::Document &d, DataSavingFunctions::ProgramState &state) {
    state.scaling_coefficient                = d["scaling_coefficient"].GetFloat();
    state.keyboard_movement_amount           = d["keyboard_movement_amount"].GetFloat();
    state.SHIFT_keyboard_movement_multiplier = d["SHIFT_keyboard_movement_multiplier"].GetFloat();

    state.font_size                        = d["font_size"].GetInt();
    state.float_precision                  = d["float_precision"].GetInt();
    state.brush_size                       = d["brush_size"].GetInt();
    state.update_info_every_n_milliseconds = d["update_info_every_n_milliseconds"].GetInt();

    state.wait_for_engine_to_stop_to_render = d["wait_for_engine_to_stop_to_render"].GetBool();
    state.disable_warnings                  = d["disable_warnings"].GetBool();
    state.really_stop_render                = d["really_stop_render"].GetBool();
    state.save_simulation_settings          = d["save_simulation_settings"].GetBool();
    state.use_point_size                    = d["use_point_size"].GetBool();

    bool use_cuda = d["use_cuda"].GetBool();
    if (cuda_is_available()) {
        state.use_cuda = use_cuda;
    }
}

void DataSavingFunctions::write_json_state(const std::string& path, ProgramState state,
                                           SimulationParameters &sp) {
    Document d;
    d.SetObject();

    write_json_version(d);
    write_json_program_settings(d, state);
    if (state.save_simulation_settings) {
        write_json_extended_simulation_parameters(d, sp);
    }

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    std::fstream file;
    file.open(path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}

bool DataSavingFunctions::read_json_state(const std::string& path, ProgramState state,
                                          SimulationParameters &sp) {
    if (!std::filesystem::exists(path)) {
        write_json_state(path, state, sp);
        return false;
    }

    std::string json;
    auto ss = std::ostringstream();
    std::ifstream file;
    file.open(path);
    if (!file.is_open()) {
        return false;
    }

    ss << file.rdbuf();
    json = ss.str();

    file.close();

    Document d;
    d.Parse(json.c_str());

    if (!read_json_version(d)) {
        return false;
    }

    read_json_program_settings(d, state);
    if (state.save_simulation_settings) { read_json_extended_simulation_parameters(d, sp); }

    return true;
}