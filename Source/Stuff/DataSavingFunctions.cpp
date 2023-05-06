//
// Created by spaceeye on 31.07.22.
//

#include "DataSavingFunctions.h"

const uint32_t SAVE_VERSION = 11;
const std::string VERSION_NAME = "main";

#define write_M(data, type, count) os.device()->write((char*)data, sizeof(type)*count)
#define read_M(data, type, count) is.readRawData((char*)data, sizeof(type)*count)

void DataSavingFunctions::write_version(QDataStream &os) {
    write_M(&SAVE_VERSION, uint32_t, 1);
    auto size = VERSION_NAME.size();
    write_M(&size, uint32_t, 1);
    write_M(VERSION_NAME.c_str(), char, VERSION_NAME.size());
}

void DataSavingFunctions::write_simulation_parameters(QDataStream & os, SimulationParameters &sp) {
    write_M(&sp, SimulationParameters, 1);
}

void DataSavingFunctions::write_organisms_block_parameters(QDataStream & os, OrganismBlockParameters &bp) {
    write_M(&bp, OrganismBlockParameters, 1);
}

void DataSavingFunctions::write_data_container_data(QDataStream & os, EngineDataContainer &edc) {
    write_M(&edc.total_engine_ticks, uint32_t, 1);
    write_M(&edc.simulation_width,   uint32_t, 1);
    write_M(&edc.simulation_height,  uint32_t, 1);
}

void DataSavingFunctions::write_simulation_grid(QDataStream & os, EngineDataContainer &edc) {
    std::vector<WorldBlocks> blocks{};

    for (uint32_t x = 0; x < edc.simulation_width; x++) {
        for (uint32_t y = 0; y < edc.simulation_height; y++) {
            auto type = edc.st_grid.get_type(x, y);

            switch (type) {
                case BlockTypes::WallBlock:
                    blocks.emplace_back(x, y, BlockTypes::WallBlock);
                    break;
                default:break;
            }
        }
    }

    uint64_t size = blocks.size();

    write_M(&size, uint64_t, 1);
    write_M(blocks.data(), WorldBlocks, blocks.size());

    size = edc.st_grid.food_vec.size();
    write_M(&size, uint64_t, 1);
    write_M(edc.st_grid.food_vec.data(), float, edc.st_grid.food_vec.size());
}

void DataSavingFunctions::write_organism(QDataStream &os, Organism *organism) {
    write_organism_brain(os,   &organism->brain);
    write_organism_anatomy(os, &organism->anatomy);
    write_organism_occ(os,      organism->occ);
    write_organism_data(os,     organism);
}

void DataSavingFunctions::write_organisms(QDataStream & os, EngineDataContainer &edc) {
    uint32_t size = edc.stc.num_alive_organisms;
    write_M(&size, uint32_t, 1);
    for (int i = 0; i <= edc.stc.last_alive_position; i++) {
        auto & organism = edc.stc.organisms[i];
        if (organism.is_dead) {continue;}
        write_organism(os, &organism);
    }
}

void DataSavingFunctions::write_organism_data(QDataStream & os, Organism * organism) {
    write_M(static_cast<OrganismData*>(organism), OrganismData, 1);
}

void DataSavingFunctions::write_organism_brain(QDataStream & os, Brain * brain) {
    write_M(brain, Brain, 1);
}

void DataSavingFunctions::write_organism_anatomy(QDataStream & os, Anatomy * anatomy) {
    uint32_t organism_blocks_size = anatomy->organism_blocks.size();
    uint32_t producing_space_size = anatomy->producing_space.size();
    uint32_t eating_space_size    = anatomy->eating_space.size();
    uint32_t killing_space_size   = anatomy->killing_space.size();
    uint32_t eye_vec_size         = anatomy->eye_block_vec.size();

    uint32_t eating_mask_size = anatomy->eating_mask.size();
    uint32_t killer_mask_size = anatomy->killer_mask.size();

    write_M(&organism_blocks_size, uint32_t, 1);
    write_M(&producing_space_size, uint32_t, 1);
    write_M(&eating_space_size,    uint32_t, 1);
    write_M(&killing_space_size,   uint32_t, 1);
    write_M(&eye_vec_size,         uint32_t, 1);

    write_M(&eating_mask_size, uint32_t, 1);
    write_M(&killer_mask_size, uint32_t, 1);

    for (auto & item: anatomy->c.data) {
        write_M(&item, int32_t, 1);
    }

    write_M(anatomy->organism_blocks.data(), SerializedOrganismBlockContainer, anatomy->organism_blocks.size());
    write_M(anatomy->eating_space.data(),    SerializedAdjacentSpaceContainer, anatomy->eating_space.size());
    write_M(anatomy->killing_space.data(),   SerializedAdjacentSpaceContainer, anatomy->killing_space.size());
    write_M(anatomy->eye_block_vec.data(),   SerializedAdjacentSpaceContainer, anatomy->eye_block_vec.size());

    write_M(anatomy->eating_mask.data(),   uint32_t, anatomy->eating_mask.size());
    write_M(anatomy->killer_mask.data(),   uint32_t, anatomy->killer_mask.size());

    for (auto & space: anatomy->producing_space) {
        auto space_size = space.size();
        write_M(&space_size, uint32_t, 1);
        write_M(space.data(), SerializedAdjacentSpaceContainer, space_size);
    }
}


bool DataSavingFunctions::read_version(QDataStream &is) {
    uint32_t save_version;
    uint32_t version_len;
    std::string version_name;

    read_M(&save_version, uint32_t, 1);
    read_M(&version_len, uint32_t, 1);
    version_name.resize(version_len, *"0");
    read_M(version_name.c_str(), char, version_len);
    return (save_version == SAVE_VERSION) && (version_name == VERSION_NAME);
}

void DataSavingFunctions::read_simulation_parameters(QDataStream &is, SimulationParameters &sp) {
    read_M(&sp, SimulationParameters, 1);
}

void DataSavingFunctions::read_organisms_block_parameters(QDataStream &is, OrganismBlockParameters &bp) {
    read_M(&bp, OrganismBlockParameters, 1);
}

void DataSavingFunctions::read_data_container_data(QDataStream &is, EngineDataContainer &edc, uint32_t &sim_width, uint32_t &sim_height) {
    read_M(&edc.loaded_engine_ticks, uint32_t, 1);
    read_M(&sim_width,               uint32_t, 1);
    read_M(&sim_height,              uint32_t, 1);
}

void DataSavingFunctions::read_simulation_grid(QDataStream &is, EngineDataContainer &edc) {
    std::vector<WorldBlocks> blocks{};
    std::uint64_t size;

    read_M(&size, uint64_t, 1);
    blocks.resize(size);
    read_M(blocks.data(), WorldBlocks, size);

    read_M(&size, uint64_t, 1);
    read_M(edc.st_grid.food_vec.data(), float, edc.st_grid.food_vec.size());

    for (auto & block: blocks) {
        edc.st_grid.get_type(block.x, block.y) = block.type;
    }
}

void DataSavingFunctions::read_organism(QDataStream &is, SimulationParameters &sp, OrganismBlockParameters &bp,
                                        Organism *organism, OCCParameters &occp, OCCLogicContainer &occl) {
    auto brain = Brain();
    auto anatomy = Anatomy();
    auto occ = OrganismConstructionCode();

    read_organism_brain(is, &brain);
    read_organism_anatomy(is, &anatomy);
    read_organism_occ(is, occ);

    *organism = Organism(0,
                         0,
                         Rotation::UP,
                         anatomy,
                         brain,
                         occ,
                         &sp,
                         &bp,
                         &occp,
                         &occl, 0, 0, 0, false);
    read_organism_data(is, *static_cast<OrganismData*>(organism));
}

//TODO save child patterns?
bool DataSavingFunctions::read_organisms(QDataStream &is, EngineDataContainer &edc, SimulationParameters &sp,
                                         OrganismBlockParameters &bp, uint32_t num_organisms, OCCParameters &occp,
                                         OCCLogicContainer &occl) {
    edc.stc.organisms.reserve(num_organisms);
    for (int i = 0; i <= num_organisms; i++) {
        auto * organism = OrganismsController::get_new_main_organism(edc);
        auto array_place = organism->vector_index;
        read_organism(is, sp, bp, organism, occp, occl);
        organism->vector_index = array_place;
        SimulationEngineSingleThread::place_organism(edc, *organism, sp, false);
    }

    return false;
}

void DataSavingFunctions::read_organism_data(QDataStream &is, OrganismData & data) {
    read_M(&data, OrganismData, 1);
}

void DataSavingFunctions::read_organism_brain(QDataStream &is, Brain * brain) {
    read_M(brain, Brain, 1);
}

void DataSavingFunctions::read_organism_anatomy(QDataStream &is, Anatomy * anatomy) {
    uint32_t organism_blocks_size = 0;
    uint32_t producing_space_size = 0;
    uint32_t eating_space_size    = 0;
    uint32_t killing_space_size   = 0;
    uint32_t eye_vec_size         = 0;

    uint32_t eating_mask_size = 0;
    uint32_t killer_mask_size = 0;

    read_M(&organism_blocks_size, uint32_t, 1);
    read_M(&producing_space_size, uint32_t, 1);
    read_M(&eating_space_size,    uint32_t, 1);
    read_M(&killing_space_size,   uint32_t, 1);
    read_M(&eye_vec_size,         uint32_t, 1);

    read_M(&eating_mask_size, uint32_t, 1);
    read_M(&killer_mask_size, uint32_t, 1);

    anatomy->organism_blocks.resize(organism_blocks_size);
    anatomy->producing_space.resize(producing_space_size);
    anatomy->eating_space   .resize(eating_space_size);
    anatomy->killing_space  .resize(killing_space_size);
    anatomy->eye_block_vec  .resize(eye_vec_size);

    anatomy->eating_mask.resize(eating_mask_size);
    anatomy->killer_mask.resize(killer_mask_size);

    for (auto & item: anatomy->c.data) {
        read_M(&item, int32_t, 1);
    }

    read_M(anatomy->organism_blocks.data(), SerializedOrganismBlockContainer, anatomy->organism_blocks.size());
    read_M(anatomy->eating_space.data(),    SerializedAdjacentSpaceContainer, anatomy->eating_space.size());
    read_M(anatomy->killing_space.data(),   SerializedAdjacentSpaceContainer, anatomy->killing_space.size());
    read_M(anatomy->eye_block_vec.data(),   SerializedAdjacentSpaceContainer, anatomy->eye_block_vec.size());

    read_M(anatomy->eating_mask.data(),   uint32_t, anatomy->eating_mask.size());
    read_M(anatomy->killer_mask.data(),   uint32_t, anatomy->killer_mask.size());

    for (auto & space: anatomy->producing_space) {
        uint32_t space_size;
        read_M(&space_size, uint32_t, 1);
        space.resize(space_size);
        read_M(space.data(), SerializedAdjacentSpaceContainer, space_size);
    }
}


const uint32_t STATE_SAVE_VERSION = 3;

using rapidjson::Value, rapidjson::Document, rapidjson::StringBuffer, rapidjson::Writer, rapidjson::kObjectType, rapidjson::kArrayType;

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

    d.AddMember("weighted_brain_mutation_step", Value(sp.weighted_brain_mutation_step), d.GetAllocator());
    d.AddMember("threshold_move",               Value(sp.threshold_move), d.GetAllocator());

    d.AddMember("continuous_movement_drag", Value(sp.continuous_movement_drag), d.GetAllocator());

    d.AddMember("food_threshold", Value(sp.food_threshold), d.GetAllocator());
    d.AddMember("max_food",       Value(sp.max_food), d.GetAllocator());

    d.AddMember("min_move_range",       Value(sp.min_move_range), d.GetAllocator());
    d.AddMember("max_move_range",       Value(sp.max_move_range), d.GetAllocator());
    d.AddMember("move_range_delimiter", Value(sp.move_range_delimiter), d.GetAllocator());
    d.AddMember("set_fixed_move_range", Value(sp.set_fixed_move_range), d.GetAllocator());
    d.AddMember("min_organism_size",    Value(sp.min_organism_size), d.GetAllocator());

    d.AddMember("starting_organism_size", Value(sp.starting_organism_size), d.GetAllocator());
    d.AddMember("cell_growth_modifier",   Value(sp.cell_growth_modifier), d.GetAllocator());
    d.AddMember("cell_growth_time",       Value(sp.cell_growth_time), d.GetAllocator());

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
    d.AddMember("no_random_decisions",              Value(sp.no_random_decisions), d.GetAllocator());
    d.AddMember("use_occ",                          Value(sp.use_occ), d.GetAllocator());
    d.AddMember("recenter_to_imaginary_pos",        Value(sp.recenter_to_imaginary_pos), d.GetAllocator());
    d.AddMember("do_not_mutate_brains_of_plants",   Value(sp.do_not_mutate_brains_of_plants), d.GetAllocator());
    d.AddMember("use_weighted_brain",               Value(sp.use_weighted_brain), d.GetAllocator());
    d.AddMember("organisms_destroy_food",           Value(sp.organisms_destroy_food), d.GetAllocator());
    d.AddMember("use_continuous_movement",          Value(sp.use_continuous_movement), d.GetAllocator());
    d.AddMember("growth_of_organisms",              Value(sp.growth_of_organisms), d.GetAllocator());
    d.AddMember("food_blocks_movement",             Value(sp.food_blocks_movement), d.GetAllocator());
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

    sp.weighted_brain_mutation_step = d["weighted_brain_mutation_step"].GetFloat();
    sp.threshold_move               = d["threshold_move"].GetFloat();

    sp.food_threshold = d["food_threshold"].GetFloat();
    sp.max_food       = d["max_food"].GetFloat();

    sp.continuous_movement_drag = d["continuous_movement_drag"].GetFloat();

    sp.min_move_range       = d["min_move_range"].GetInt();
    sp.max_move_range       = d["max_move_range"].GetInt();
    sp.move_range_delimiter = d["move_range_delimiter"].GetFloat();
    sp.set_fixed_move_range = d["set_fixed_move_range"].GetBool();
    sp.min_organism_size    = d["min_organism_size"].GetInt();

    sp.starting_organism_size = d["starting_organism_size"].GetInt();
    sp.cell_growth_modifier   = d["cell_growth_modifier"].GetFloat();
    sp.cell_growth_time       = d["cell_growth_time"].GetInt();

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
    sp.no_random_decisions              = d["no_random_decisions"].GetBool();
    sp.use_occ                          = d["use_occ"].GetBool();
    sp.recenter_to_imaginary_pos        = d["recenter_to_imaginary_pos"].GetBool();
    sp.do_not_mutate_brains_of_plants   = d["do_not_mutate_brains_of_plants"].GetBool();
    sp.use_weighted_brain               = d["use_weighted_brain"].GetBool();
    sp.organisms_destroy_food           = d["organisms_destroy_food"].GetBool();
    sp.use_continuous_movement          = d["use_continuous_movement"].GetBool();
    sp.growth_of_organisms              = d["growth_of_organisms"].GetBool();
    sp.food_blocks_movement             = d["food_blocks_movement"].GetBool();
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

    state.disable_warnings                  = d["disable_warnings"].GetBool();
    state.really_stop_render                = d["really_stop_render"].GetBool();
    state.save_simulation_settings          = d["save_simulation_settings"].GetBool();
    state.use_point_size                    = d["use_point_size"].GetBool();

    bool use_cuda = d["use_cuda"].GetBool();
    if (cuda_is_available()) {
        state.use_cuda = use_cuda;
    }
}

void DataSavingFunctions::write_json_state(const std::string &path, ProgramState state, SimulationParameters &sp,
                                           OCCParameters &occp) {
    Document d;
    d.SetObject();

    write_json_version(d);
    write_json_program_settings(d, state);
    if (state.save_simulation_settings) {
        write_json_extended_simulation_parameters(d, sp);
        write_json_occp(d, occp);
    }

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    std::fstream file;
    file.open(path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}

void DataSavingFunctions::read_json_state(const std::string &path, ProgramState state, SimulationParameters &sp,
                                          OCCParameters &occp) {

    std::function<void(const std::string *, ProgramState, SimulationParameters *, OCCParameters *)> func = &read_json_state_private;

    //if during loading function crashes (incompatible settings), then overwrite them with current version of setting.
    if (try_and_catch_abort(func, &path, state, &sp, &occp)) {
        write_json_state(path, state, sp, occp);
    }
}

void DataSavingFunctions::read_json_state_private(const std::string *path_, ProgramState state,
                                                  SimulationParameters *sp_, OCCParameters *occp_) {
    auto & path = *path_;
    auto & sp = *sp_;
    auto & occp = *occp_;

    if (!std::filesystem::exists(path)) {
        write_json_state(path, state, sp, occp);
        return;
    }

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

    Document d;
    d.Parse(json.c_str());

    if (!read_json_version(d)) {
        return;
    }

    read_json_program_settings(d, state);
    if (state.save_simulation_settings) {
        read_json_extended_simulation_parameters(d, sp);
        read_json_occp(d, occp);
    }
}

void DataSavingFunctions::write_occp(QDataStream &os, OCCParameters &occp) {
    write_M(&occp.uniform_mutation_distribution, bool, 1);
    write_M(&occp.uniform_group_size_distribution, bool, 1);
    write_M(&occp.uniform_occ_instructions_mutation, bool, 1);
    write_M(&occp.uniform_swap_distance, bool, 1);

    uint32_t size = occp.mutation_type_weights.size();
    write_M(&size, uint32_t, 1);
    write_M(occp.mutation_type_weights.data(), uint32_t, size);

    size = occp.max_group_size;
    write_M(&size, uint32_t, 1);
    write_M(occp.group_size_weights.data(), uint32_t, size);

    size = occp.occ_instructions_mutation_weights.size();
    write_M(&size, uint32_t, 1);
    write_M(occp.occ_instructions_mutation_weights.data(), uint32_t, size);

    size = occp.max_distance;
    write_M(&size, uint32_t, 1);
    write_M(occp.swap_distance_mutation_weights.data(), uint32_t, size);
}

void DataSavingFunctions::read_occp(QDataStream &is, OCCParameters &occp) {
    read_M(&occp.uniform_mutation_distribution, bool, 1);
    read_M(&occp.uniform_group_size_distribution, bool, 1);
    read_M(&occp.uniform_occ_instructions_mutation, bool, 1);
    read_M(&occp.uniform_swap_distance, bool, 1);

    uint32_t size;
    read_M(&size, uint32_t, 1);
    read_M(occp.mutation_type_weights.data(), uint32_t, size);

    read_M(&size, uint32_t, 1);
    occp.max_group_size = size;
    occp.group_size_weights.resize(occp.max_group_size);
    read_M(occp.group_size_weights.data(), uint32_t, size);

    read_M(&size, uint32_t, 1);
    if (size != occp.occ_instructions_mutation_weights.size()) {
        throw std::runtime_error("empty instruction mutation weights.");
    }
    read_M(occp.occ_instructions_mutation_weights.data(), uint32_t, size);

    read_M(&size, uint32_t, 1);
    occp.max_distance = size;
    read_M(occp.swap_distance_mutation_weights.data(), uint32_t, size);

    occp.mutation_discrete_distribution = std::discrete_distribution<int>{occp.mutation_type_weights.begin(), occp.mutation_type_weights.end()};
    occp.group_size_discrete_distribution = std::discrete_distribution<int>{occp.group_size_weights.begin(), occp.group_size_weights.end()};
    occp.occ_instructions_mutation_discrete_distribution = std::discrete_distribution<int>{occp.occ_instructions_mutation_weights.begin(), occp.occ_instructions_mutation_weights.end()};
    occp.swap_distance_mutation_discrete_distribution = std::discrete_distribution<int>{occp.swap_distance_mutation_weights.begin(), occp.swap_distance_mutation_weights.end()};
}

void DataSavingFunctions::write_json_occp(Document & d, OCCParameters & occp) {
    d.AddMember("uniform_mutation_distribution",     Value(occp.uniform_mutation_distribution), d.GetAllocator());
    d.AddMember("uniform_group_size_distribution",   Value(occp.uniform_group_size_distribution), d.GetAllocator());
    d.AddMember("uniform_occ_instructions_mutation", Value(occp.uniform_occ_instructions_mutation), d.GetAllocator());
    d.AddMember("uniform_swap_distance",             Value(occp.uniform_swap_distance), d.GetAllocator());

    Value mtw(kArrayType);
    for (auto & value: occp.mutation_type_weights) {
        mtw.PushBack(Value(value), d.GetAllocator());
    }
    d.AddMember("mutation_type_weights", mtw, d.GetAllocator());

    Value gsw(kArrayType);
    for (auto & value: occp.group_size_weights) {
        gsw.PushBack(Value(value), d.GetAllocator());
    }
    d.AddMember("group_size_weights", gsw, d.GetAllocator());

    Value imw(kArrayType);
    for (auto & value: occp.occ_instructions_mutation_weights) {
        imw.PushBack(Value(value), d.GetAllocator());
    }
    d.AddMember("occ_instructions_mutation_weights", imw, d.GetAllocator());

    Value mdw(kArrayType);
    for (auto & value: occp.swap_distance_mutation_weights) {
        mdw.PushBack(Value(value), d.GetAllocator());
    }
    d.AddMember("swap_distance_mutation_weights", mdw, d.GetAllocator());
}

void DataSavingFunctions::read_json_occp(Document & d, OCCParameters & occp) {
    occp.uniform_mutation_distribution     = d["uniform_mutation_distribution"].GetBool();
    occp.uniform_group_size_distribution   = d["uniform_group_size_distribution"].GetBool();
    occp.uniform_occ_instructions_mutation = d["uniform_occ_instructions_mutation"].GetBool();
    occp.uniform_swap_distance             = d["uniform_swap_distance"].GetBool();

    int i = 0;
    std::vector<int> temp_weights{};
    temp_weights.clear();
    for (auto & value: d["mutation_type_weights"].GetArray()) {
        occp.mutation_type_weights[i] = value.GetInt();
        i++;
    }

    temp_weights.clear();
    for (auto & value: d["group_size_weights"].GetArray()) {
        temp_weights.emplace_back(value.GetInt());
    }
    occp.group_size_weights = std::vector(temp_weights);
    occp.max_group_size = temp_weights.size();

    temp_weights.clear();
    for (auto & value: d["occ_instructions_mutation_weights"].GetArray()) {
        temp_weights.emplace_back(value.GetInt());
    }
    occp.occ_instructions_mutation_weights = std::vector(temp_weights);

    temp_weights.clear();
    for (auto & value: d["swap_distance_mutation_weights"].GetArray()) {
        temp_weights.emplace_back(value.GetInt());
    }
    occp.swap_distance_mutation_weights = std::vector(temp_weights);
    occp.max_distance = temp_weights.size();

    occp.mutation_discrete_distribution = std::discrete_distribution<int>{occp.mutation_type_weights.begin(), occp.mutation_type_weights.end()};
    occp.group_size_discrete_distribution = std::discrete_distribution<int>{occp.group_size_weights.begin(), occp.group_size_weights.end()};
    occp.occ_instructions_mutation_discrete_distribution = std::discrete_distribution<int>{occp.occ_instructions_mutation_weights.begin(), occp.occ_instructions_mutation_weights.end()};
    occp.swap_distance_mutation_discrete_distribution = std::discrete_distribution<int>{occp.swap_distance_mutation_weights.begin(), occp.swap_distance_mutation_weights.end()};
}

void DataSavingFunctions::write_organism_occ(QDataStream &os, OrganismConstructionCode &occ) {
    int size = occ.get_code_const_ref().size();
    write_M(&size, int32_t , 1);
    if (size > 0) {
        write_M(occ.get_code_const_ref().data(), OCCInstruction, size);
    }
}

void DataSavingFunctions::read_organism_occ(QDataStream &is, OrganismConstructionCode &occ) {
    int size = 0;
    read_M(&size, int32_t, 1);
    occ.get_code_ref().resize(size);
    if (size > 0) {
        read_M(occ.get_code_ref().data(), OCCInstruction, size);
    }
}









void write_json_data(QDataStream &stream, EngineDataContainer &edc, SimulationParameters &sp, double total_total_mutation_rate) {
//    Document d;
//    d.SetObject();
//
//    d.AddMember("num_rows", Value(edc.simulation_height - 2), d.GetAllocator());
//    d.AddMember("num_cols", Value(edc.simulation_width - 2), d.GetAllocator());
//    d.AddMember("total_mutability", Value(int(total_total_mutation_rate*100)), d.GetAllocator());
//    d.AddMember("largest_cell_count", Value(0), d.GetAllocator());
//    d.AddMember("reset_count", Value(0), d.GetAllocator());
//    d.AddMember("total_ticks", Value(edc.total_engine_ticks), d.GetAllocator());
//    d.AddMember("data_update_rate", Value(100), d.GetAllocator());
//
//    json_write_grid(d, edc);
//    json_write_organisms(d, edc, sp);
//    json_write_fossil_record(d);
//    json_write_controls(d, sp);
//
//    StringBuffer buffer;
//    Writer<StringBuffer> writer(buffer);
//    d.Accept(writer);
//
//    std::fstream file;
//    file.open(path, std::ios_base::out);
//    file << buffer.GetString();
//    file.close();
}

void DataSavingFunctions::json_write_grid(Document & d, EngineDataContainer &edc) {
    Value j_grid(kObjectType);
    j_grid.AddMember("cols", Value(edc.simulation_width  - 2), d.GetAllocator());
    j_grid.AddMember("rows", Value(edc.simulation_height - 2), d.GetAllocator());

    Value food(kArrayType);
    Value walls(kArrayType);

    for (int x = 1; x < edc.simulation_width - 1; x++) {
        for (int y = 1; y < edc.simulation_height - 1; y++) {
            auto type = edc.st_grid.get_type(x, y);
            if (type != BlockTypes::WallBlock && type != BlockTypes::FoodBlock) {continue;}
            Value cell(kObjectType);

            cell.AddMember("c", Value(x-1), d.GetAllocator());
            cell.AddMember("r", Value(y-1), d.GetAllocator());

            if (type == BlockTypes::FoodBlock) {
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
//    j_anatomy.AddMember("is_producer",    Value(static_cast<bool>(organism->anatomy._producer_blocks)), d.GetAllocator());
//    j_anatomy.AddMember("is_mover",       Value(static_cast<bool>(organism->anatomy._mover_blocks)), d.GetAllocator());
//    j_anatomy.AddMember("has_eyes",       Value(static_cast<bool>(organism->anatomy._eye_blocks)), d.GetAllocator());

    for (auto & block: organism->anatomy.organism_blocks) {
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
//    decisions.AddMember("food",     Value(static_cast<int>(table.FoodBlock)),     d.GetAllocator());
//    decisions.AddMember("wall",     Value(static_cast<int>(table.WallBlock)),     d.GetAllocator());
//    decisions.AddMember("mouth",    Value(static_cast<int>(table.MouthBlock)),    d.GetAllocator());
//    decisions.AddMember("producer", Value(static_cast<int>(table.ProducerBlock)), d.GetAllocator());
//    decisions.AddMember("mover",    Value(static_cast<int>(table.MoverBlock)),    d.GetAllocator());
//    decisions.AddMember("killer",   Value(static_cast<int>(table.KillerBlock)),   d.GetAllocator());
//    decisions.AddMember("armor",    Value(static_cast<int>(table.ArmorBlock)),    d.GetAllocator());
//    decisions.AddMember("eye",      Value(static_cast<int>(table.EyeBlock)),      d.GetAllocator());

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
//        table.FoodBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["food"]    .GetInt());
//        table.WallBlock     = static_cast<SimpleDecision>(organism["brain"]["decisions"]["wall"]    .GetInt());
//        table.MouthBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mouth"]   .GetInt());
//        table.ProducerBlock = static_cast<SimpleDecision>(organism["brain"]["decisions"]["producer"].GetInt());
//        table.MoverBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["mover"]   .GetInt());
//        table.KillerBlock   = static_cast<SimpleDecision>(organism["brain"]["decisions"]["killer"]  .GetInt());
//        table.ArmorBlock    = static_cast<SimpleDecision>(organism["brain"]["decisions"]["armor"]   .GetInt());
//        table.EyeBlock      = static_cast<SimpleDecision>(organism["brain"]["decisions"]["eye"]     .GetInt());
    }

    if (rotation == 1) {rotation = 3;}
    else if (rotation == 3) {rotation = 1;}

    *new_organism = Organism(x,
                             y,
                             static_cast<Rotation>(rotation),
                             anatomy,
                             brain, OrganismConstructionCode(),
                             &sp,
                             &bp, nullptr, nullptr,
                             move_range,
                             mutability, 0, false);
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

        if (new_organism->anatomy.c["mover"] > 0 && new_organism->anatomy.c["eye"] > 0) { new_organism->brain.brain_type = BrainTypes::SimpleBrain;}

        SimulationEngineSingleThread::place_organism(edc, *new_organism, sp);
    }

    auto gen = lehmer64(42);

    for (auto & organism: edc.stc.organisms) {
        edc.stc.organisms_observations.clear();
        SimulationEngineSingleThread::reserve_observations(edc.stc.organisms_observations, edc.stc.organisms, edc);
        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {SimulationEngineSingleThread::get_observations(edc, sp, organism, edc.stc.organisms_observations);}}

        for (int i = 0; i <= edc.stc.last_alive_position; i++) {auto & organism = edc.stc.organisms[i]; if (!organism.is_dead) {organism.think_decision(edc.stc.organisms_observations[i], gen);}}
    }
}
