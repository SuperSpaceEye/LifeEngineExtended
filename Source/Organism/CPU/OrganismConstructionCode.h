//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H
#define LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H

#include "Anatomy.h"
#include "OrganismConstructionCodeInstruction.h"
#include "../../Containers/CPU/OrganismConstructionCodeParameters.h"

//https://github.com/DavidPal/discrete-distribution
//https://stackoverflow.com/questions/53632441/c-sampling-from-discrete-distribution-without-replacement

class OrganismConstructionCode {
public:
    OrganismConstructionCode()=default;
    OrganismConstructionCode(const OrganismConstructionCode & parent_code);
    OrganismConstructionCode(OrganismConstructionCode && code_to_move);

    OrganismConstructionCode && mutate(OCCParameters & occp, lehmer64 & gen);
    SerializedOrganismStructureContainer * compile_code();

private:
    std::vector<OCCInstruction> occ_vector;
    //min x, max x, min y, max y
    std::array<int, 4> calculate_construction_edges();

    std::vector<SerializedOrganismBlockContainer> && compile_base_structure();
    std::vector<std::vector<SerializedAdjacentSpaceContainer>> && compile_producing_space();
    std::vector<SerializedAdjacentSpaceContainer> && compile_eating_space();
    std::vector<SerializedAdjacentSpaceContainer> && compile_killing_space();

};


#endif //LIFEENGINEEXTENDED_ORGANISMCONSTRUCTIONCODE_H
