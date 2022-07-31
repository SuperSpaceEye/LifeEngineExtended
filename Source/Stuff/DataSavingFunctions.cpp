//
// Created by spaceeye on 31.07.22.
//

#include "DataSavingFunctions.h"

//TODO increment every time saving logic changes
uint32_t SAVE_VERSION = 5;

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

void DataSavingFunctions::write_organisms(std::ofstream & os, EngineDataContainer &edc) {
    uint32_t size = edc.organisms.size();
    os.write((char*)&size, sizeof(uint32_t));
    for (auto & organism: edc.organisms) {
        write_organism_brain(os,   &organism->brain);
        write_organism_anatomy(os, &organism->anatomy);
        write_organism_data(os,    organism);
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

//TODO save child patterns?
bool DataSavingFunctions::read_organisms(std::ifstream& is, EngineDataContainer &edc, SimulationParameters &sp, OrganismBlockParameters &bp, uint32_t num_organisms) {
    edc.organisms.reserve(num_organisms);
    for (int i = 0; i < num_organisms; i++) {
        auto brain = Brain();
        auto anatomy = Anatomy();

        read_organism_brain(is, &brain);
        read_organism_anatomy(is, &anatomy);

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