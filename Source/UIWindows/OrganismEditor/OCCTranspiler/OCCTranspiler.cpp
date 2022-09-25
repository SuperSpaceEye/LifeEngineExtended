//
// Created by spaceeye on 25.09.22.
//

#include "OCCTranspiler.h"

OCCTranspilingErrorCodes OCCTranspiler::transpile(std::string &&code) {
    std::string instruction{};
    for (int i = 0; i < code.size(); i++) {
        char ch = code[i];
        character++;
        //if space or new line then do nothing
        if (ch == " "[0]) {}
        else if (ch == "\n"[0]) {
            line++;
            character = 0;
        }

        //if start of a comment
        else if (ch == "/"[0]) {
            //cycle the code until new line
            while (i < code.size()) {
                ch = code[i];
                if (ch == "\n"[0]) {
                    break;
                }
                i++;
            }
        }

        else if (ch == ";"[0]) {
            auto err = check_and_add_instruction(instruction);
            if (err != OCCTranspilingErrorCodes::NoError) {
                return err;
            }
            instruction = "";
        } else {
            instruction += ch;
        }
    }

    if (transpiled_instructions.empty()) { return OCCTranspilingErrorCodes::NoInstructionsAfterTranspiling;}

    return OCCTranspilingErrorCodes::NoError;
}

OCCTranspilingErrorCodes OCCTranspiler::check_and_add_instruction(std::string &instruction_name) {
    if (instruction_name.empty()) { return OCCTranspilingErrorCodes::NoError;}

    auto instruction_index = get_index_of_occ_instruction(instruction_name);
    if (instruction_index != -1) {
        transpiled_instructions.emplace_back(static_cast<OCCInstruction>(instruction_index));
        return OCCTranspilingErrorCodes::NoError;
    }

    instruction_index = get_index_of_occ_instruction_short(instruction_name);
    if (instruction_index != -1) {
        transpiled_instructions.emplace_back(static_cast<OCCInstruction>(instruction_index));
        return OCCTranspilingErrorCodes::NoError;
    }

    unknown_instruction = instruction_name;
    return OCCTranspilingErrorCodes::UnknownInstruction;
}

std::vector<OCCInstruction> OCCTranspiler::get_transpiled_instructions() {
    auto temp = std::vector(transpiled_instructions);
    transpiled_instructions.clear();
    line = 0;
    character = 0;
    unknown_instruction = "";
    return temp;
}
