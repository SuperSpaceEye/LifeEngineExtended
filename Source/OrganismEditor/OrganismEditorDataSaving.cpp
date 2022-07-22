//
// Created by spaceeye on 21.07.22.
//

#include "OrganismEditor.h"

//TODO remove duplicate code.

template <typename T>
struct my_id_translator
{
    typedef T internal_type;
    typedef T external_type;

    boost::optional<T> get_value(const T &v) { return  v.substr(1, v.size() - 2) ; }
    boost::optional<T> put_value(const T &v) { return '"' + v +'"'; }
};

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

void OrganismEditor::read_json_organism(std::string &full_path) {
    boost::property_tree::ptree root;
    boost::property_tree::json_parser::read_json(full_path, root);

    auto brain = std::make_shared<Brain>();
    auto anatomy = std::make_shared<Anatomy>();

    int y              = root.get<int>("r")+1;
    int x              = root.get<int>("c")+1;
    int rotation       = root.get<int>("rotation");
    int move_range     = root.get<int>("move_range");
    float mutability   = float(root.get<float>("mutability"))/100;
    bool is_mover      = root.get<bool>("anatomy.is_mover");
    bool has_eyes      = root.get<bool>("anatomy.has_eyes");

    auto block_data = std::vector<SerializedOrganismBlockContainer>{};

    for (auto & cell: root.get_child("anatomy.cells")) {
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
        table.FoodBlock     = static_cast<SimpleDecision>(root.get<int>("brain.decisions.food"));
        table.WallBlock     = static_cast<SimpleDecision>(root.get<int>("brain.decisions.wall"));
        table.MouthBlock    = static_cast<SimpleDecision>(root.get<int>("brain.decisions.mouth"));
        table.ProducerBlock = static_cast<SimpleDecision>(root.get<int>("brain.decisions.producer"));
        table.MoverBlock    = static_cast<SimpleDecision>(root.get<int>("brain.decisions.mover"));
        table.KillerBlock   = static_cast<SimpleDecision>(root.get<int>("brain.decisions.killer"));
        table.ArmorBlock    = static_cast<SimpleDecision>(root.get<int>("brain.decisions.armor"));
        table.EyeBlock      = static_cast<SimpleDecision>(root.get<int>("brain.decisions.eye"));
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
    boost::property_tree::ptree cell, anatomy, cells, j_organism, brain;

    j_organism = boost::property_tree::ptree{};
    anatomy = boost::property_tree::ptree{};
    cells = boost::property_tree::ptree{};
    brain = boost::property_tree::ptree{};

    j_organism.put("c", editor_organism->x-1);
    j_organism.put("r", editor_organism->y-1);
    j_organism.put("lifetime", editor_organism->lifetime);
    j_organism.put("food_collected", static_cast<int>(editor_organism->food_collected));
    j_organism.put("living", true);
    j_organism.put("direction", 2);
    j_organism.put("rotation", static_cast<int>(editor_organism->rotation));
    j_organism.put("can_rotate", editor_organism->sp->runtime_rotation_enabled);
    j_organism.put("move_count", 0);
    j_organism.put("move_range", editor_organism->move_range);
    j_organism.put("ignore_brain_for", 0);
    j_organism.put("mutability", static_cast<int>(editor_organism->anatomy_mutation_rate*100));
    j_organism.put("damage", editor_organism->damage);

    anatomy.put("birth_distance", 6);
    anatomy.put("is_producer", static_cast<bool>(editor_organism->anatomy->_producer_blocks));
    anatomy.put("is_mover", static_cast<bool>(editor_organism->anatomy->_mover_blocks));
    anatomy.put("has_eyes", static_cast<bool>(editor_organism->anatomy->_eye_blocks));

    for (auto & block: editor_organism->anatomy->_organism_blocks) {
        cell = boost::property_tree::ptree{};
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

    auto & table = editor_organism->brain->simple_action_table;

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

    std::fstream file;
    file.open(full_path, std::ios_base::out);
    boost::property_tree::json_parser::write_json(file, j_organism);
    file.close();
}

namespace pt = boost::property_tree;namespace pt = boost::property_tree;