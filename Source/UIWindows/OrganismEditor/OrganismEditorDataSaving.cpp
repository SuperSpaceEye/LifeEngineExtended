//
// Created by spaceeye on 21.07.22.
//

#include "OrganismEditor.h"

void OrganismEditor::read_organism(std::ifstream &is) {
    OCCParameters p{};
    OCCLogicContainer l{};

    DataSavingFunctions::read_organism(is, *sp, *editor_organism.bp, chosen_organism, p, l);

    load_chosen_organism();
}

using rapidjson::Document, rapidjson::StringBuffer, rapidjson::Writer;

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

    file.close();

    Document organism;
    organism.Parse(json.c_str());

    if (!organism.HasMember("r")) {
        display_message("Failed to load organism");
        return;
    }

    DataSavingFunctions::json_read_organism(organism, *sp, *editor_organism.bp, chosen_organism);

    load_chosen_organism();
}

void OrganismEditor::write_json_organism(std::string &full_path) {
    Document j_organism;
    j_organism.SetObject();

    DataSavingFunctions::write_json_organism(j_organism, &editor_organism, j_organism, *sp);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    j_organism.Accept(writer);

    std::fstream file;
    file.open(full_path, std::ios_base::out);
    file << buffer.GetString();
    file.close();
}