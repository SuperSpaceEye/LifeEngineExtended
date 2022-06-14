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
    os.write((char*)(&sp), sizeof(SimulationParameters));
}

void WindowCore::write_organisms_block_parameters(std::ofstream& os) {
    os.write((char*)(&bp), sizeof(BlockParameters));
}

void WindowCore::write_data_container_data(std::ofstream& os) {
    os.write((char*)(&dc.total_engine_ticks), sizeof(uint32_t));
    os.write((char*)(&dc.simulation_width),   sizeof(uint16_t));
    os.write((char*)(&dc.simulation_height),  sizeof(uint16_t));
}
//    void WindowCore::write_color_container(){}
void WindowCore::write_simulation_grid(std::ofstream& os) {
    for (auto & col: dc.CPU_simulation_grid) {
        os.write((char*)(&col[0]), sizeof(AtomicGridBlock)*col.size());
    }
}
void WindowCore::write_organisms(std::ofstream& os) {
    uint32_t size = dc.organisms.size();
    os.write((char*)(&size), sizeof(uint32_t));
    for (auto & organism: dc.organisms) {
        write_organism_data(os, organism);
        write_organism_brain(os, organism->brain.get());
        write_organism_anatomy(os, organism->organism_anatomy.get());
    }
}

void WindowCore::write_organism_data(std::ofstream& os, Organism * organism) {
    os.write((char*)(organism), sizeof(Organism));
}

void WindowCore::write_organism_brain(std::ofstream& os, Brain * brain) {
    os.write((char*)(brain), sizeof(Brain));
}

void WindowCore::write_organism_anatomy(std::ofstream& os, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = anatomy->_organism_blocks.size();
    uint32_t producing_space_size                = anatomy->_producing_space.size();
    uint32_t eating_space_size                   = anatomy->_eating_space.size();
    uint32_t killing_space_size                  = anatomy->_killing_space.size();
    uint32_t single_adjacent_space_size          = anatomy->_single_adjacent_space.size();
    uint32_t single_diagonal_adjacent_space_size = anatomy->_single_diagonal_adjacent_space.size();
    uint32_t double_adjacent_space_size          = anatomy->_double_adjacent_space.size();

    os.write((char*)(&organism_blocks_size),                sizeof(uint32_t));
    os.write((char*)(&producing_space_size),                sizeof(uint32_t));
    os.write((char*)(&eating_space_size),                   sizeof(uint32_t));
    os.write((char*)(&killing_space_size),                  sizeof(uint32_t));
    os.write((char*)(&single_adjacent_space_size),          sizeof(uint32_t));
    os.write((char*)(&single_diagonal_adjacent_space_size), sizeof(uint32_t));
    os.write((char*)(&double_adjacent_space_size),          sizeof(uint32_t));

    os.write((char*)(&anatomy->_mouth_blocks),    sizeof(int32_t));
    os.write((char*)(&anatomy->_producer_blocks), sizeof(int32_t));
    os.write((char*)(&anatomy->_mover_blocks),    sizeof(int32_t));
    os.write((char*)(&anatomy->_killer_blocks),   sizeof(int32_t));
    os.write((char*)(&anatomy->_armor_blocks),    sizeof(int32_t));
    os.write((char*)(&anatomy->_eye_blocks),      sizeof(int32_t));

    os.write((char*)(&anatomy->_organism_blocks[0]),                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    os.write((char*)(&anatomy->_eating_space[0]),                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    os.write((char*)(&anatomy->_killing_space[0]),                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());
    os.write((char*)(&anatomy->_single_adjacent_space[0]),          sizeof(SerializedArmorSpaceContainer   ) * anatomy->_single_adjacent_space.size());
    os.write((char*)(&anatomy->_single_diagonal_adjacent_space[0]), sizeof(SerializedAdjacentSpaceContainer) * anatomy->_single_diagonal_adjacent_space.size());
    os.write((char*)(&anatomy->_double_adjacent_space[0]),          sizeof(SerializedAdjacentSpaceContainer) * anatomy->_double_adjacent_space.size());

    for (auto & space: anatomy->_producing_space) {
        auto size = space.size();
        os.write((char*)(&size), sizeof(uint32_t));
        os.write((char*)(&space[0]), sizeof(SerializedAdjacentSpaceContainer) * size);
    }
}


void WindowCore::read_data(std::ifstream &is) {
    for (auto & organism: dc.organisms) {
        delete organism;
    }
    dc.organisms.clear();
    for (auto & organism: dc.to_place_organisms) {
        delete organism;
    }
    dc.to_place_organisms.clear();

    read_simulation_parameters(is);
    read_organisms_block_parameters(is);
    read_data_container_data(is);
    read_simulation_grid(is);
    read_organisms(is);
}

void WindowCore::read_simulation_parameters(std::ifstream& is) {
    is.read((char*)(&sp), sizeof(SimulationParameters));
}

void WindowCore::read_organisms_block_parameters(std::ifstream& is) {
    is.read((char*)(&bp), sizeof(BlockParameters));
}

void WindowCore::read_data_container_data(std::ifstream& is) {
    is.read((char*)(&dc.total_engine_ticks), sizeof(uint32_t));
    is.read((char*)(&dc.simulation_width),   sizeof(uint16_t));
    is.read((char*)(&dc.simulation_height),  sizeof(uint16_t));
}
//    void WindowCore::read_color_container(){}
void WindowCore::read_simulation_grid(std::ifstream& is) {
    disable_warnings = true;
    resize_simulation_space();
    disable_warnings = false;

    for (auto & col: dc.CPU_simulation_grid) {
        is.read((char*)(&col[0]), sizeof(AtomicGridBlock)*col.size());
    }
}

void WindowCore::read_organisms(std::ifstream& is) {
    uint32_t num_organisms;
    is.read((char*)(&num_organisms), sizeof(uint32_t));
    dc.organisms.reserve(num_organisms);
    for (int i = 0; i < num_organisms; i++) {
        std::cout << num_organisms << " " << i << "\n";
        auto * organism = new Organism();
        auto brain = std::make_shared<Brain>();
        auto anatomy = std::make_shared<Anatomy>();

        dc.organisms.emplace_back(organism);

        read_organism_data(is, organism);
        read_organism_brain(is, brain.get());
        read_organism_anatomy(is, anatomy.get());

        organism->brain = brain;
        organism->organism_anatomy = anatomy;
        organism->bp = &bp;
        organism->sp = &sp;
        organism->child_pattern = nullptr;
        organism->child_ready = false;

        SimulationEngineSingleThread::place_organism(&dc, organism);
    }
}

void WindowCore::read_organism_data(std::ifstream& is, Organism * organism) {
    is.read((char*)(organism), sizeof(Organism));
}

void WindowCore::read_organism_brain(std::ifstream& is, Brain * brain) {
    is.read((char*)(brain), sizeof(Brain));
}

void WindowCore::read_organism_anatomy(std::ifstream& is, Anatomy * anatomy) {
    uint32_t organism_blocks_size                = 0;
    uint32_t producing_space_size                = 0;
    uint32_t eating_space_size                   = 0;
    uint32_t killing_space_size                  = 0;
    uint32_t single_adjacent_space_size          = 0;
    uint32_t single_diagonal_adjacent_space_size = 0;
    uint32_t double_adjacent_space_size          = 0;

    is.read((char*)(&organism_blocks_size),                sizeof(uint32_t));
    is.read((char*)(&producing_space_size),                sizeof(uint32_t));
    is.read((char*)(&eating_space_size),                   sizeof(uint32_t));
    is.read((char*)(&killing_space_size),                  sizeof(uint32_t));
    is.read((char*)(&single_adjacent_space_size),          sizeof(uint32_t));
    is.read((char*)(&single_diagonal_adjacent_space_size), sizeof(uint32_t));
    is.read((char*)(&double_adjacent_space_size),          sizeof(uint32_t));

    anatomy->_organism_blocks               .reserve(organism_blocks_size);
    anatomy->_producing_space               .reserve(producing_space_size);
    anatomy->_eating_space                  .reserve(eating_space_size);
    anatomy->_killing_space                 .reserve(killing_space_size);
    anatomy->_single_adjacent_space         .reserve(single_adjacent_space_size);
    anatomy->_single_diagonal_adjacent_space.reserve(single_diagonal_adjacent_space_size);
    anatomy->_double_adjacent_space         .reserve(double_adjacent_space_size);

    is.read((char*)(&anatomy->_mouth_blocks),    sizeof(int32_t));
    is.read((char*)(&anatomy->_producer_blocks), sizeof(int32_t));
    is.read((char*)(&anatomy->_mover_blocks),    sizeof(int32_t));
    is.read((char*)(&anatomy->_killer_blocks),   sizeof(int32_t));
    is.read((char*)(&anatomy->_armor_blocks),    sizeof(int32_t));
    is.read((char*)(&anatomy->_eye_blocks),      sizeof(int32_t));

    is.read((char*)(&anatomy->_organism_blocks[0]),                sizeof(SerializedOrganismBlockContainer) * anatomy->_organism_blocks.size());
    is.read((char*)(&anatomy->_eating_space[0]),                   sizeof(SerializedAdjacentSpaceContainer) * anatomy->_eating_space.size());
    is.read((char*)(&anatomy->_killing_space[0]),                  sizeof(SerializedAdjacentSpaceContainer) * anatomy->_killing_space.size());
    is.read((char*)(&anatomy->_single_adjacent_space[0]),          sizeof(SerializedArmorSpaceContainer   ) * anatomy->_single_adjacent_space.size());
    is.read((char*)(&anatomy->_single_diagonal_adjacent_space[0]), sizeof(SerializedAdjacentSpaceContainer) * anatomy->_single_diagonal_adjacent_space.size());
    is.read((char*)(&anatomy->_double_adjacent_space[0]),          sizeof(SerializedAdjacentSpaceContainer) * anatomy->_double_adjacent_space.size());

    for (auto & space: anatomy->_producing_space) {
        uint32_t space_size;
        is.read((char*)(&space_size), sizeof(uint32_t));
        is.read((char*)(&space[0]), sizeof(SerializedAdjacentSpaceContainer) * space_size);
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
//            _ui.table_organism_block_parameters->update();
        }
    }
    _ui.table_organism_block_parameters->update();
}
