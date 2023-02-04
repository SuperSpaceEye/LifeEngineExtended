//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H

#include "Anatomy.h"
#include "OrganismConstructionCodeInstruction.h"
#include "../../Containers/CPU/OrganismConstructionCodeParameters.h"
#include "../../Containers/CPU/OCCLogicContainer.h"
#include "AnatomyContainers.h"

//https://github.com/DavidPal/discrete-distribution
//https://stackoverflow.com/questions/53632441/c-sampling-from-discrete-distribution-without-replacement

enum class OCCMutations {
    AppendRandom,
    InsertRandom,
    ChangeRandom,
    DeleteRandom,
    SwapRandom
};

class OrganismConstructionCode {
public:
    OrganismConstructionCode()=default;
    OrganismConstructionCode(const OrganismConstructionCode & parent_code);
//    OrganismConstructionCode(OrganismConstructionCode && code_to_move) noexcept {occ_vector = std::move(code_to_move.occ_vector);};
    OrganismConstructionCode & operator=(const OrganismConstructionCode & code) {occ_vector = std::vector(code.occ_vector); return *this;}
    OrganismConstructionCode & operator=(OrganismConstructionCode && code)  noexcept {occ_vector = std::move(code.occ_vector); return *this;}

    OrganismConstructionCode mutate(OCCParameters & occp, lehmer64 & gen);
    SerializedOrganismStructureContainer * compile_code(OCCLogicContainer & occ_container);

    void set_code(std::vector<OCCInstruction> && code) {occ_vector = std::move(code);}
    const std::vector<OCCInstruction> & get_code_const_ref() {return occ_vector;}
    std::vector<OCCInstruction> & get_code_ref() {return occ_vector;}
private:
    std::vector<OCCInstruction> occ_vector;
    //min x, max x, min y, max y
    std::array<int, 4> calculate_construction_edges();

    std::vector<SerializedOrganismBlockContainer>
    compile_base_structure(SerializedOrganismStructureContainer *container, OCCLogicContainer &occ_c, const std::array<int, 4> &edges);
    //producing space, eating space, killing space
    static SerializedOrganismStructureContainer *
    compile_spaces(OCCLogicContainer &occ_c, const std::array<int, 4> &edges,
                   std::vector<SerializedOrganismBlockContainer> &organism_blocks,
                   SerializedOrganismStructureContainer *container);

    static void shift_instruction_part(SerializedOrganismStructureContainer *container, OCCLogicContainer &occ_c,
                                       const std::array<std::array<int, 2>, 8> &shift_values, std::array<int, 2> &shift,
                                       const Rotation base_rotation, int cursor_x, int cursor_y, int center_x, int center_y,
                                       const OCCInstruction instruction, const OCCInstruction next_instruction,
                                       std::vector<SerializedOrganismBlockContainer> &blocks, int &i, bool &pass) ;
};


#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H
