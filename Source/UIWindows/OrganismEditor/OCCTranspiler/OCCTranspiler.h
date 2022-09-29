//
// Created by spaceeye on 25.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCTRANSPILER_H
#define LIFEENGINEEXTENDED_OCCTRANSPILER_H

#include <vector>
#include <string>

#include "../../../Organism/CPU/OrganismConstructionCodeInstruction.h"

enum class OCCTranspilingErrorCodes {
    NoError,
    UnknownInstruction,
    NoInstructionsAfterTranspiling,
    FirstInstructionNotSetBlock
};

class OCCTranspiler {
private:
    std::vector<OCCInstruction> transpiled_instructions{};

    OCCTranspilingErrorCodes check_and_add_instruction(std::string & instruction_name);
public:
    int line = 0;
    int character = 0;
    std::string unknown_instruction{};

    OCCTranspiler()=default;
    OCCTranspilingErrorCodes transpile(std::string && code);
    std::vector<OCCInstruction> get_transpiled_instructions();
    static std::string convert_to_text_code(const std::vector<OCCInstruction> & instructions, bool short_instructions);
};


#endif //LIFEENGINEEXTENDED_OCCTRANSPILER_H
