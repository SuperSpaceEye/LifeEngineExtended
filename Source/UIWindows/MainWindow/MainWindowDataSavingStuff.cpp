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
    DataSavingFunctions::write_occp(os, occp);
    DataSavingFunctions::write_data_container_data(os, edc);
    DataSavingFunctions::write_simulation_grid(os, edc);
    DataSavingFunctions::write_organisms(os, edc);
}

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
    DataSavingFunctions::read_occp(is, occp);
    if (read_data_container_data(is)) {
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
    }
    read_simulation_grid(is);
    if (read_organisms(is)) {
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
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

    DataSavingFunctions::read_organisms(is, edc, sp, bp, num_organisms, occp, edc.stc.occl);
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

#include <csetjmp>
#include <csignal>

jmp_buf env;

void on_sigabrt (int signum)
{
    signal (signum, SIG_DFL);
    longjmp (env, 1);
}

template <typename ...A>
bool try_and_catch_abort(std::function<void(A...)> f, A... args)
{
    if (setjmp (env) == 0) {
        signal(SIGABRT, &on_sigabrt);
        f(args...);
        signal (SIGABRT, SIG_DFL);
        return false;
    }
    else {
        return true;
    }
}

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

    SimulationParameters recovery_sp = sp;
    OrganismBlockParameters recovery_bp = bp;
    uint32_t recovery_simulation_width = edc.simulation_width;
    uint32_t recovery_simulation_height = edc.simulation_height;

    std::function<void(rapidjson::GenericDocument<rapidjson::UTF8<>>*, int32_t*, int32_t*)> func1 = &json_read_sim_width_height;
    if (try_and_catch_abort(func1, reinterpret_cast<rapidjson::GenericDocument<rapidjson::UTF8<>>*>(&document), &edc.simulation_width, &edc.simulation_height)) {
        display_message("Failed to read grids width or height.");
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
    }
    json_resize_and_make_walls(document);
    std::function<void(rapidjson::GenericDocument<rapidjson::UTF8<>>*, EngineDataContainer*)> func2 = &json_read_ticks_food_walls;
    if (try_and_catch_abort(func2, reinterpret_cast<rapidjson::GenericDocument<rapidjson::UTF8<>>*>(&document), &edc)) {
        display_message("Failed to read positions of walls or food.");
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
    }

    std::function<void(rapidjson::GenericDocument<rapidjson::UTF8<>>*, SimulationParameters*)> func3 = &DataSavingFunctions::json_read_simulation_parameters;
    if (try_and_catch_abort(func3, reinterpret_cast<rapidjson::GenericDocument<rapidjson::UTF8<>>*>(&document), &sp)) {
        display_message("Failed to read simulation parameters");
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
    }

    std::function<void(rapidjson::GenericDocument<rapidjson::UTF8<>>*, SimulationParameters*, OrganismBlockParameters*, EngineDataContainer*)> func4 = &DataSavingFunctions::json_read_organisms_data;
    if (try_and_catch_abort(func4, reinterpret_cast<rapidjson::GenericDocument<rapidjson::UTF8<>>*>(&document), &sp, &bp, &edc)) {
        display_message("Failed to read simulation parameters");
        recover_state(recovery_sp, recovery_bp, recovery_simulation_width, recovery_simulation_height);
        return;
    }

    edc.total_engine_ticks = edc.loaded_engine_ticks;
}

void MainWindow::json_read_sim_width_height(Document * d_, int32_t * new_width, int32_t * new_height) {
    auto & d = *d_;
    *new_height = d["grid"]["rows"].GetInt() + 2;
    *new_width  = d["grid"]["cols"].GetInt() + 2;
}

void MainWindow::json_resize_and_make_walls(Document &d) {
    new_simulation_width = edc.simulation_width;
    new_simulation_height = edc.simulation_height;

    fill_window = false;
    ui.cb_fill_window->setChecked(false);
    disable_warnings = true;

    update_simulation_size_label();

    just_resize_simulation_grid();

    engine.make_walls();
    disable_warnings = false;
}

void MainWindow::json_read_ticks_food_walls(Document *d_, EngineDataContainer *edc_) {
    auto & d = *d_;
    auto & edc = *edc_;

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
