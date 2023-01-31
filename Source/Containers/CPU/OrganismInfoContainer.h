//
// Created by spaceeye on 06.08.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H
#define LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H

#include "EngineDataContainer.h"
#include "EngineControlParametersContainer.h"
#include "../../Organism/CPU/Organism.h"
#include "../../Stuff/BlockTypes.hpp"

struct OrganismInfoHolder {
    double size = 0;
    double _organism_lifetime = 0;
    double _organism_age    = 0;
    double _gathered_food   = 0;
    double brain_mutation_rate = 0;
    double anatomy_mutation_rate = 0;
    double occ_instructions_num = 0;
    int64_t total_occ_instructions_num = 0;
    int64_t total = 0;
    std::array<double, NUM_ORGANISM_BLOCKS> block_avgs{};
};
struct OrganismInfoContainer {
    int64_t total_size_organism_blocks = 0;
    int64_t total_size_producing_space = 0;
    int64_t total_size_eating_space    = 0;
    int64_t total_size_single_adjacent_space = 0;
    int64_t total_size_single_diagonal_adjacent_space = 0;
    int64_t total_size = 0;

    //0 - total avg, 1 - station avg, 2 - moving avg
    std::array<OrganismInfoHolder, 3> avgs{};

    double move_range = 0;
    int64_t moving_organisms = 0;
    int64_t organisms_with_eyes = 0;

    [[nodiscard]] const OrganismInfoContainer & get_info() const {
        return *this;
    }

    void parse_info(EngineDataContainer * edc, EngineControlParameters * ecp) {
        *this = OrganismInfoContainer{};
        parse_organisms_info(*this, edc, ecp);
    }

private:
    static void parse_organisms_info(OrganismInfoContainer & info, EngineDataContainer * edc, EngineControlParameters * ecp) {
        for (auto & organism: edc->stc.organisms) {
            if (organism.is_dead) { continue;}

            info.total_size_organism_blocks += organism.anatomy.organism_blocks.size();
            info.total_size_producing_space += organism.anatomy.producing_space.size();
            info.total_size_eating_space    += organism.anatomy.eating_space.size();

            if (organism.anatomy.c["mover"] > 0) {
                info.move_range += organism.move_range;
                info.moving_organisms++;

                if(organism.anatomy.c["eye"] > 0) {
                    info.organisms_with_eyes++;
                }
            }

            for (int n = 0; n < 3; n++) {
                if (
                  !(n == 0
                || (n == 1 && organism.anatomy.c["mover"] == 0)
                || (n == 2 && organism.anatomy.c["mover"] > 0))) {continue;}

                info.avgs[n].size += organism.anatomy.organism_blocks.size();

                info.avgs[n]._organism_lifetime += organism.max_lifetime;
                info.avgs[n]._organism_age      += organism.lifetime;
                info.avgs[n]._gathered_food     += organism.food_collected;
                for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
                    info.avgs[n].block_avgs[i] += organism.anatomy.c.data[i];
                }
                info.avgs[n].total_occ_instructions_num += organism.occ.get_code_const_ref().size();

                info.avgs[n].brain_mutation_rate   += organism.brain_mutation_rate;
                info.avgs[n].anatomy_mutation_rate += organism.anatomy_mutation_rate;
                info.avgs[n].total++;
            }
        }

        info.total_size_organism_blocks                *= sizeof(SerializedOrganismBlockContainer);
        info.total_size_producing_space                *= sizeof(SerializedAdjacentSpaceContainer);
        info.total_size_eating_space                   *= sizeof(SerializedAdjacentSpaceContainer);
        info.total_size_single_adjacent_space          *= sizeof(SerializedAdjacentSpaceContainer);
        info.total_size_single_diagonal_adjacent_space *= sizeof(SerializedAdjacentSpaceContainer);

        info.move_range /= info.moving_organisms;

        info.total_size = info.total_size_organism_blocks +
                          info.total_size_producing_space +
                          info.total_size_eating_space +
                          info.total_size_single_adjacent_space +
                          info.total_size_single_diagonal_adjacent_space +
                          (sizeof(Brain) * info.avgs[0].total) +
                          (sizeof(Anatomy) * info.avgs[0].total) +
                          (sizeof(Organism) * info.avgs[0].total);

        for (int n = 0; n < 3; n++) {
            info.avgs[n].size /= info.avgs[n].total;

            info.avgs[n]._organism_lifetime /= info.avgs[n].total;
            info.avgs[n]._organism_age      /= info.avgs[n].total;
            info.avgs[n]._gathered_food     /= info.avgs[n].total;
            for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
                info.avgs[n].block_avgs[i] /= info.avgs[n].total;
            }
            info.avgs[n].occ_instructions_num = info.avgs[n].total_occ_instructions_num ? (double)info.avgs[n].total_occ_instructions_num / info.avgs[n].total : 0;

            info.avgs[n].brain_mutation_rate   /= info.avgs[n].total;
            info.avgs[n].anatomy_mutation_rate /= info.avgs[n].total;

            if (std::isnan(info.avgs[n].size))               {info.avgs[n].size               = 0;}

            if (std::isnan(info.avgs[n]._organism_lifetime)) {info.avgs[n]._organism_lifetime = 0;}
            if (std::isnan(info.avgs[n]._organism_age))      {info.avgs[n]._organism_age      = 0;}
            if (std::isnan(info.avgs[n]._gathered_food))     {info.avgs[n]._gathered_food     = 0;}
            for (int i = 0; i < NUM_ORGANISM_BLOCKS; i++) {
                if (std::isnan(info.avgs[n].block_avgs[i])) { info.avgs[n].block_avgs[i] = 0; }
            }
            if (std::isnan(info.avgs[n].occ_instructions_num)) {info.avgs[n].occ_instructions_num = 0;}

            if (std::isnan(info.avgs[n].brain_mutation_rate))   {info.avgs[n].brain_mutation_rate   = 0;}
            if (std::isnan(info.avgs[n].anatomy_mutation_rate)) {info.avgs[n].anatomy_mutation_rate = 0;}
        }
    }
};
#endif //LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H
