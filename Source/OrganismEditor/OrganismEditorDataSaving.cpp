//
// Created by spaceeye on 21.07.22.
//

#include "OrganismEditor.h"

//TODO remove duplicate code.

void OrganismEditor::read_organism(std::ifstream &is) {
    auto brain = std::make_shared<Brain>();
    auto anatomy = std::make_shared<Anatomy>();

    read_organism_brain(is, brain.get());
    read_organism_anatomy(is, anatomy.get());

    auto * organism = new Organism(0,
                                   0,
                                   Rotation::UP,
                                   anatomy,
                                   brain,
                                   editor_organism->sp,
                                   editor_organism->bp,
                                   0,
                                   0,
                                   0);

    read_organism_data(is, *static_cast<OrganismData*>(organism));

    delete editor_organism;
    *chosen_organism = organism;
    load_chosen_organism();
}

void OrganismEditor::read_organism_data(std::ifstream& is, OrganismData & data) {
    is.read((char*)&data, sizeof(OrganismData));
}

void OrganismEditor::read_organism_brain(std::ifstream& is, Brain * brain) {
    is.read((char*)brain, sizeof(Brain));
}

void OrganismEditor::read_organism_anatomy(std::ifstream& is, Anatomy * anatomy) {
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

void OrganismEditor::write_organism(std::ofstream& os) {
    write_organism_brain(os,   editor_organism->brain.get());
    write_organism_anatomy(os, editor_organism->anatomy.get());
    write_organism_data(os,    editor_organism);
}

void OrganismEditor::write_organism_data(std::ofstream& os, Organism * organism) {
    os.write((char*)static_cast<OrganismData*>(organism), sizeof(OrganismData));
}

void OrganismEditor::write_organism_brain(std::ofstream& os, Brain * brain) {
    os.write((char*)brain, sizeof(Brain));
}

void OrganismEditor::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
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

using rapidjson::Value, rapidjson::Document, rapidjson::StringBuffer, rapidjson::Writer, rapidjson::kObjectType, rapidjson::kArrayType;

void OrganismEditor::read_json_organism(std::string &full_path) {
    std::string json;
    auto ss = std::ostringstream();
    std::ifstream file;
    file.open(full_path);
    if (!file.is_open()) {
        return;
    }

    ss << file.rdbuf();
    json = ss.str();

    Document organism;
    organism.Parse(json.c_str());

    auto brain = std::make_shared<Brain>();
    auto anatomy = std::make_shared<Anatomy>();

    int y                = organism["r"].GetInt()+1;
    int x                = organism["c"].GetInt()+1;
    int rotation         = organism["rotation"].GetInt();
    int move_range       = organism["move_range"].GetInt();
    float mutability     = organism["mutability"].GetFloat()/100;
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
                                       editor_organism->sp,
                                       editor_organism->bp,
                                       move_range,
                                       mutability);
    delete editor_organism;
    *chosen_organism = new_organism;
    load_chosen_organism();
}

void OrganismEditor::write_json_organism(std::string &full_path) {
    Document j_organism;
    j_organism.SetObject();

    Value j_anatomy(kObjectType);
    Value j_brain(kObjectType);
    Value cells(kArrayType);

    j_organism.AddMember("c",                Value(editor_organism->x-1), j_organism.GetAllocator());
    j_organism.AddMember("r",                Value(editor_organism->y-1), j_organism.GetAllocator());
    j_organism.AddMember("lifetime",         Value(editor_organism->lifetime), j_organism.GetAllocator());
    j_organism.AddMember("food_collected",   Value(editor_organism->food_collected), j_organism.GetAllocator());
    j_organism.AddMember("living",           Value(true), j_organism.GetAllocator());
    j_organism.AddMember("direction",        Value(2), j_organism.GetAllocator());
    j_organism.AddMember("rotation",         Value(static_cast<int>(editor_organism->rotation)), j_organism.GetAllocator());
    j_organism.AddMember("can_rotate",             Value(editor_organism->sp->runtime_rotation_enabled), j_organism.GetAllocator());
    j_organism.AddMember("move_count",       Value(0), j_organism.GetAllocator());
    j_organism.AddMember("move_range",       Value(editor_organism->move_range), j_organism.GetAllocator());
    j_organism.AddMember("ignore_brain_for", Value(0), j_organism.GetAllocator());
    j_organism.AddMember("mutability",       Value(editor_organism->anatomy_mutation_rate*100), j_organism.GetAllocator());
    j_organism.AddMember("damage",           Value(editor_organism->damage), j_organism.GetAllocator());

    j_anatomy.AddMember("birth_distance", Value(6), j_organism.GetAllocator());
    j_anatomy.AddMember("is_producer",    Value(static_cast<bool>(editor_organism->anatomy->_producer_blocks)), j_organism.GetAllocator());
    j_anatomy.AddMember("is_mover",       Value(static_cast<bool>(editor_organism->anatomy->_mover_blocks)), j_organism.GetAllocator());
    j_anatomy.AddMember("has_eyes",       Value(static_cast<bool>(editor_organism->anatomy->_eye_blocks)), j_organism.GetAllocator());

    for (auto & block: editor_organism->anatomy->_organism_blocks) {
        Value cell(kObjectType);
        std::string state_name;

        cell.AddMember("loc_col", Value(block.relative_x), j_organism.GetAllocator());
        cell.AddMember("loc_row", Value(block.relative_y), j_organism.GetAllocator());

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
            cell.AddMember("direction", Value(static_cast<int>(block.rotation)), j_organism.GetAllocator());
        }

        Value state(kObjectType);
        state.AddMember("name", Value(state_name.c_str(), state_name.length(), j_organism.GetAllocator()), j_organism.GetAllocator());

        cell.AddMember("state", state, j_organism.GetAllocator());

        cells.PushBack(cell, j_organism.GetAllocator());
    }
    j_anatomy.AddMember("cells", cells, j_organism.GetAllocator());

    j_organism.AddMember("anatomy", j_anatomy, j_organism.GetAllocator());

    auto & table = editor_organism->brain->simple_action_table;

    Value decisions(kObjectType);

    decisions.AddMember("empty",    Value(0), j_organism.GetAllocator());
    decisions.AddMember("food",     Value(static_cast<int>(table.FoodBlock)),     j_organism.GetAllocator());
    decisions.AddMember("wall",     Value(static_cast<int>(table.WallBlock)),     j_organism.GetAllocator());
    decisions.AddMember("mouth",    Value(static_cast<int>(table.MouthBlock)),    j_organism.GetAllocator());
    decisions.AddMember("producer", Value(static_cast<int>(table.ProducerBlock)), j_organism.GetAllocator());
    decisions.AddMember("mover",    Value(static_cast<int>(table.MoverBlock)),    j_organism.GetAllocator());
    decisions.AddMember("killer",   Value(static_cast<int>(table.KillerBlock)),   j_organism.GetAllocator());
    decisions.AddMember("armor",    Value(static_cast<int>(table.ArmorBlock)),    j_organism.GetAllocator());
    decisions.AddMember("eye",      Value(static_cast<int>(table.EyeBlock)),      j_organism.GetAllocator());

    j_brain.AddMember("decisions", decisions, j_organism.GetAllocator());

    j_organism.AddMember("brain", j_brain, j_organism.GetAllocator());

    j_organism.AddMember("species_name", Value("0000000000"), j_organism.GetAllocator());

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    j_organism.Accept(writer);

    std::fstream file;
    file.open(full_path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}