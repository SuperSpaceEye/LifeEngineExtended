//
// Created by spaceeye on 31.07.22.
//

#ifndef LIFEENGINEEXTENDED_DATASAVINGFUNCTIONS_H
#define LIFEENGINEEXTENDED_DATASAVINGFUNCTIONS_H

#include <cstdint>
#include <ostream>
#include <fstream>
#include <filesystem>

#include "Stuff/external/rapidjson/document.h"
#include "Stuff/external/rapidjson/writer.h"
#include "Stuff/external/rapidjson/stringbuffer.h"

#include "Containers/SimulationParameters.h"
#include "Containers/EngineDataContainer.h"
#include "Containers/OrganismBlockParameters.h"
#include "Containers/OrganismConstructionCodeParameters.h"
#include "Organism/Organism.h"
#include "Organism/Anatomy/Anatomy.h"
#include "Organism/Brain/Brain.h"
#include "SimulationEngine/SimulationEngineModes/SimulationEngineSingleThread.h"
#include "SimulationEngine/OrganismsController.h"

#include "Stuff/enums/BlockTypes.hpp"
#include "UIMisc.h"

namespace DataSavingFunctions {
    struct WorldBlocks {
        uint32_t x;
        uint32_t y;
        BlockTypes type;
        WorldBlocks()=default;
        WorldBlocks(uint32_t x, uint32_t y, BlockTypes type): x(x), y(y), type(type){}
    };

    struct ProgramState {
        float & scaling_coefficient;
        float & keyboard_movement_amount;
        float & SHIFT_keyboard_movement_multiplier;

        int & font_size;
        int & float_precision;
        int & brush_size;
        int & update_info_every_n_milliseconds;

        bool & use_cuda;
        bool & disable_warnings;
        bool & really_stop_render;
        bool & save_simulation_settings;
        bool & use_point_size;
    };

    void write_version(std::ofstream &os);
    void write_simulation_parameters(std::ofstream & os, SimulationParameters &sp);
    void write_organisms_block_parameters(std::ofstream & os, OrganismBlockParameters &bp);
    void write_occp(std::ofstream & os, OCCParameters & occp);
    void write_data_container_data(std::ofstream & os, EngineDataContainer &edc);
    void write_simulation_grid(std::ofstream & os, EngineDataContainer &edc);
    void write_organism(std::ofstream& os, Organism * organism);
    void write_organisms(std::ofstream & os, EngineDataContainer &edc);
    void write_organism_data(std::ofstream & os, Organism * organism);
    void write_organism_brain(std::ofstream & os, Brain * brain);
    void write_organism_anatomy(std::ofstream & os, Anatomy * anatomy);
    void write_organism_occ(std::ofstream& os, OrganismConstructionCode & occ);

    bool read_version(std::ifstream &is);
    void read_simulation_parameters(std::ifstream& is, SimulationParameters &sp);
    void read_organisms_block_parameters(std::ifstream& is, OrganismBlockParameters &bp);
    void read_data_container_data(std::ifstream& is, EngineDataContainer &edc, uint32_t &sim_width, uint32_t &sim_height);
    void read_occp(std::ifstream& is, OCCParameters & occp);
    void read_simulation_grid(std::ifstream& is, EngineDataContainer &edc);
    void read_organism(std::ifstream &is, SimulationParameters &sp, OrganismBlockParameters &bp,
                       Organism *organism, OCCParameters &occp, OCCLogicContainer &occl);
    bool read_organisms(std::ifstream &is, EngineDataContainer &edc, SimulationParameters &sp,
                        OrganismBlockParameters &bp, uint32_t num_organisms, OCCParameters &occp,
                        OCCLogicContainer &occl);
    void read_organism_data(std::ifstream& is, OrganismData & data);
    void read_organism_brain(std::ifstream& is, Brain * brain);
    void read_organism_anatomy(std::ifstream& is, Anatomy * anatomy);
    void read_organism_occ(std::ifstream& is, OrganismConstructionCode & occ);


    void write_json_data(const std::string &path, EngineDataContainer &edc, SimulationParameters &sp, double total_total_mutation_rate);
    void json_write_grid(rapidjson::Document &d, EngineDataContainer &edc);
    void json_write_organisms(rapidjson::Document &d, EngineDataContainer &edc, SimulationParameters &sp);
    void write_json_organism(rapidjson::Document &d, Organism * organism, rapidjson::Value &j_organism, SimulationParameters &sp);
    void json_write_fossil_record(rapidjson::Document &d);
    void json_write_controls(rapidjson::Document &d, SimulationParameters &sp);

//    void json_read_simulation_parameters(rapidjson::Document & d, SimulationParameters &sp);
    void json_read_organism(rapidjson::GenericValue<rapidjson::UTF8<>> &organism, SimulationParameters &sp,
                            OrganismBlockParameters &bp, Organism *new_organism);
    void json_read_organisms_data(rapidjson::Document *d_, SimulationParameters *sp_, OrganismBlockParameters *bp_, EngineDataContainer *edc_);

    void json_read_simulation_parameters(rapidjson::Document * d, SimulationParameters * sp);

    void write_json_version(rapidjson::Document & d);
    bool read_json_version(rapidjson::Document & d);

    void write_json_extended_simulation_parameters(rapidjson::Document & d, SimulationParameters &sp);
    void read_json_extended_simulation_parameters(rapidjson::Document & d, SimulationParameters &sp);

    void write_json_program_settings(rapidjson::Document & d, DataSavingFunctions::ProgramState &state);
    void read_json_program_settings(rapidjson::Document & d, DataSavingFunctions::ProgramState &state);

    void write_json_state(const std::string &path, ProgramState state, SimulationParameters &sp,
                          OCCParameters &occp);
    void read_json_state(const std::string &path, ProgramState state, SimulationParameters &sp,
                         OCCParameters &occp);

    void write_json_occp(rapidjson::Document & d, OCCParameters & parameters);
    void read_json_occp(rapidjson::Document & d, OCCParameters & parameters);

    void read_json_state_private(const std::string *path_, ProgramState state, SimulationParameters *sp_,
                                 OCCParameters *occp_);
}

#endif //LIFEENGINEEXTENDED_DATASAVINGFUNCTIONS_H
