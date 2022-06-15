//
// Created by spaceeye on 13.06.22.
//

#include <memory>

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

void WindowCore::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = anatomy->_organism_blocks.size();
    uint32_t producing_space_size                = anatomy->_producing_space.size();
    uint32_t eating_space_size                   = anatomy->_eating_space.size();
    uint32_t killing_space_size                  = anatomy->_killing_space.size();
    uint32_t single_adjacent_space_size          = anatomy->_single_adjacent_space.size();
    uint32_t single_diagonal_adjacent_space_size = anatomy->_single_diagonal_adjacent_space.size();
    uint32_t double_adjacent_space_size          = anatomy->_double_adjacent_space.size();

    os.write((char*)&organism_blocks_size,                sizeof(uint32_t));
    os.write((char*)&producing_space_size,                sizeof(uint32_t));
    os.write((char*)&eating_space_size,                   sizeof(uint32_t));
    os.write((char*)&killing_space_size,                  sizeof(uint32_t));
    os.write((char*)&single_adjacent_space_size,          sizeof(uint32_t));
    os.write((char*)&single_diagonal_adjacent_space_size, sizeof(uint32_t));
    os.write((char*)&double_adjacent_space_size,          sizeof(uint32_t));

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
    os.write((char*)&anatomy->_double_adjacent_space[0],          sizeof(SerializedAdjacentSpaceContainer) * anatomy->_double_adjacent_space.size());

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
        organism->child_ready             = false;
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
    uint32_t double_adjacent_space_size          = 0;

    is.read((char*)&organism_blocks_size,                sizeof(uint32_t));
    is.read((char*)&producing_space_size,                sizeof(uint32_t));
    is.read((char*)&eating_space_size,                   sizeof(uint32_t));
    is.read((char*)&killing_space_size,                  sizeof(uint32_t));
    is.read((char*)&single_adjacent_space_size,          sizeof(uint32_t));
    is.read((char*)&single_diagonal_adjacent_space_size, sizeof(uint32_t));
    is.read((char*)&double_adjacent_space_size,          sizeof(uint32_t));

    anatomy->_organism_blocks               .resize(organism_blocks_size);
    anatomy->_producing_space               .resize(producing_space_size);
    anatomy->_eating_space                  .resize(eating_space_size);
    anatomy->_killing_space                 .resize(killing_space_size);
    anatomy->_single_adjacent_space         .resize(single_adjacent_space_size);
    anatomy->_single_diagonal_adjacent_space.resize(single_diagonal_adjacent_space_size);
    anatomy->_double_adjacent_space         .resize(double_adjacent_space_size);

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
    is.read((char*)&anatomy->_double_adjacent_space[0],          sizeof(SerializedAdjacentSpaceContainer) * anatomy->_double_adjacent_space.size());

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

void WindowCore::write_json_data(std::string path) {

}

//https://www.cochoy.fr/boost-property-tree/
void WindowCore::read_json_data(std::string path) {
    pt::ptree root;
    pt::read_json(path, root);
    dc.simulation_height = root.get<int>("num_rows")+2;
    dc.simulation_width = root.get<int>("num_cols")+2;
    dc.total_engine_ticks = root.get<int>("total_ticks");

    new_simulation_width = dc.simulation_width;
    new_simulation_height = dc.simulation_height;

    fill_window = false;
    _ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    resize_simulation_space();
    make_walls();
    disable_warnings = false;

    auto test = root.get_child("grid.food");

    for (auto & pair: root.get_child("grid.food")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        dc.CPU_simulation_grid[x][y].type = BlockTypes::FoodBlock;
    }

    for (auto & pair: root.get_child("grid.walls")) {
        int y = pair.second.get<int>("r")+1;
        int x = pair.second.get<int>("c")+1;
        dc.CPU_simulation_grid[x][y].type = BlockTypes::WallBlock;
    }

    sp.lifespan_multiplier = root.get<int>("controls.lifespanMultiplier");
    sp.food_production_probability = float(root.get<int>("controls.foodProdProb"))/100;
    sp.use_anatomy_evolved_mutation_rate   = !root.get<bool>("controls.useGlobalMutability");
    sp.global_anatomy_mutation_rate = float(root.get<int>("controls.globalMutability"))/100;
    sp.add_cell = root.get<int>("controls.addProb");
    sp.change_cell = root.get<int>("controls.changeProb");
    sp.remove_cell = root.get<int>("controls.removeProb");
    sp.runtime_rotation_enabled = root.get<bool>("controls.rotationEnabled");
    sp.food_blocks_reproduction = root.get<bool>("controls.foodBlocksReproduction");
    sp.movers_can_produce_food = root.get<bool>("controls.moversCanProduce");
    sp.on_touch_kill = root.get<bool>("controls.instaKill");
    sp.look_range = root.get<int>("controls.lookRange");
//    sp.auto_produce_n_food = root.get<int>("controls.foodDropProb");
//     = root.get<int>("controls.extraMoverFoodCost");

    for (auto & organism: root.get_child("organisms")) {
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        int y = organism.second.get<int>("r")+1;
        int x = organism.second.get<int>("c")+1;
        int lifetime = organism.second.get<int>("lifetime");
        int food_collected = organism.second.get<int>("food_collected");
        int rotation = organism.second.get<int>("rotation");
        int move_range = organism.second.get<int>("move_range");
        float mutability = float(organism.second.get<int>("mutability"))/100;
        int damage = organism.second.get<int>("damage");
        bool is_mover = organism.second.get<bool>("anatomy.is_mover");
        bool has_eyes = organism.second.get<bool>("anatomy.has_eyes");
        for (auto & j_anatomy: organism.second.get_child("anatomy.cells")) {
            int l_y = j_anatomy.second.get<int>("loc_row");
            int l_x = j_anatomy.second.get<int>("loc_col");
            auto state = j_anatomy.second.get<std::string>("state.name");
            Rotation rotation = Rotation::UP;
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
                rotation = static_cast<Rotation>(j_anatomy.second.get<int>("direction"));
            } else if (state == "armor") {
                type = BlockTypes::ArmorBlock;
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

    }

//    exit(0);
}