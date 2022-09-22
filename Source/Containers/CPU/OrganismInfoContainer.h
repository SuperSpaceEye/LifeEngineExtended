//
// Created by spaceeye on 06.08.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H
#define LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H

#include "EngineDataContainer.h"
#include "EngineControlParametersContainer.h"
#include "../../Organism/CPU/Organism.h"

struct OrganismInfoHolder {
    double size = 0;
    double _organism_lifetime = 0;
    double _organism_age    = 0;
    double _gathered_food   = 0;
    double _mouth_blocks    = 0;
    double _producer_blocks = 0;
    double _mover_blocks    = 0;
    double _killer_blocks   = 0;
    double _armor_blocks    = 0;
    double _eye_blocks      = 0;
    double brain_mutation_rate = 0;
    double anatomy_mutation_rate = 0;
    double occ_instructions_num = 0;
    int64_t total_occ_instructions_num = 0;
    int64_t total = 0;
};
struct OrganismInfoContainer {
    int64_t total_size_organism_blocks = 0;
    int64_t total_size_producing_space = 0;
    int64_t total_size_eating_space    = 0;
    int64_t total_size_single_adjacent_space = 0;
    int64_t total_size_single_diagonal_adjacent_space = 0;
    int64_t total_size = 0;

    OrganismInfoHolder total_avg{};
    OrganismInfoHolder station_avg{};
    OrganismInfoHolder moving_avg{};

    double move_range = 0;
    int64_t moving_organisms = 0;
    int64_t organisms_with_eyes = 0;

    double total_total_mutation_rate = 0;

    [[nodiscard]] const OrganismInfoContainer & get_info() const {
        return *this;
    }

    void parse_info(EngineDataContainer * edc, EngineControlParameters * ecp) {
        *this = OrganismInfoContainer{};
        parse_organisms_info(*this, edc, ecp);
    }

private:
    //TODO possible optimization - iterative info updates.
    static void parse_organisms_info(OrganismInfoContainer & info, EngineDataContainer * edc, EngineControlParameters * ecp) {
        bool has_pool = true;
        int i = 0;
        while (has_pool) {
            std::vector<Organism> * pool;

            if (ecp->simulation_mode == SimulationModes::CPU_Single_Threaded) {
                pool = &edc->stc.organisms;
                has_pool = false;
            } else if (ecp->simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
//                pool = &edc->organisms_pools[i];
//                i++;
//                if (i >= ecp->num_threads) {
//                    has_pool = false;
//                }
            } else {
                throw "no pool";
            }

            for (auto & organism: *pool) {
                if (organism.is_dead) { continue;}

                info.total_size_organism_blocks += organism.anatomy._organism_blocks.size();
                info.total_size_producing_space += organism.anatomy._producing_space.size();
                info.total_size_eating_space    += organism.anatomy._eating_space.size();

                if (organism.anatomy._mover_blocks > 0) {
                    info.move_range += organism.move_range;
                    info.moving_organisms++;

                    if (organism.anatomy._eye_blocks > 0) {
                        info.organisms_with_eyes++;
                    }
                }

                info.total_avg.size += organism.anatomy._organism_blocks.size();

                info.total_avg._organism_lifetime += organism.max_lifetime;
                info.total_avg._organism_age      += organism.lifetime;
                info.total_avg._gathered_food     += organism.food_collected;
                info.total_avg._mouth_blocks      += organism.anatomy._mouth_blocks;
                info.total_avg._producer_blocks   += organism.anatomy._producer_blocks;
                info.total_avg._mover_blocks      += organism.anatomy._mover_blocks;
                info.total_avg._killer_blocks     += organism.anatomy._killer_blocks;
                info.total_avg._armor_blocks      += organism.anatomy._armor_blocks;
                info.total_avg._eye_blocks        += organism.anatomy._eye_blocks;
                info.total_avg.total_occ_instructions_num += organism.occ.get_code_const_ref().size();

                info.total_avg.brain_mutation_rate   += organism.brain_mutation_rate;
                info.total_avg.anatomy_mutation_rate += organism.anatomy_mutation_rate;
                info.total_avg.total++;

                if (organism.anatomy._mover_blocks > 0) {
                    info.moving_avg.size += organism.anatomy._organism_blocks.size();

                    info.moving_avg._organism_lifetime += organism.max_lifetime;
                    info.moving_avg._organism_age      += organism.lifetime;
                    info.moving_avg._gathered_food     += organism.food_collected;
                    info.moving_avg._mouth_blocks      += organism.anatomy._mouth_blocks;
                    info.moving_avg._producer_blocks   += organism.anatomy._producer_blocks;
                    info.moving_avg._mover_blocks      += organism.anatomy._mover_blocks;
                    info.moving_avg._killer_blocks     += organism.anatomy._killer_blocks;
                    info.moving_avg._armor_blocks      += organism.anatomy._armor_blocks;
                    info.moving_avg._eye_blocks        += organism.anatomy._eye_blocks;
                    info.moving_avg.total_occ_instructions_num += organism.occ.get_code_const_ref().size();

                    info.moving_avg.brain_mutation_rate   += organism.brain_mutation_rate;
                    info.moving_avg.anatomy_mutation_rate += organism.anatomy_mutation_rate;
                    info.moving_avg.total++;
                } else {
                    info.station_avg.size += organism.anatomy._organism_blocks.size();

                    info.station_avg._organism_lifetime += organism.max_lifetime;
                    info.station_avg._organism_age      += organism.lifetime;
                    info.station_avg._gathered_food     += organism.food_collected;
                    info.station_avg._mouth_blocks      += organism.anatomy._mouth_blocks;
                    info.station_avg._producer_blocks   += organism.anatomy._producer_blocks;
                    info.station_avg._mover_blocks      += organism.anatomy._mover_blocks;
                    info.station_avg._killer_blocks     += organism.anatomy._killer_blocks;
                    info.station_avg._armor_blocks      += organism.anatomy._armor_blocks;
                    info.station_avg._eye_blocks        += organism.anatomy._eye_blocks;
                    info.station_avg.total_occ_instructions_num += organism.occ.get_code_const_ref().size();

                    info.station_avg.brain_mutation_rate   += organism.brain_mutation_rate;
                    info.station_avg.anatomy_mutation_rate += organism.anatomy_mutation_rate;
                    info.station_avg.total++;
                }
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
                          (sizeof(Brain) * info.total_avg.total) +
                          (sizeof(Anatomy) * info.total_avg.total) +
                          (sizeof(Organism) * info.total_avg.total);

        info.total_total_mutation_rate = info.total_avg.anatomy_mutation_rate;

        info.total_avg.size /= info.total_avg.total;

        info.total_avg._organism_lifetime /= info.total_avg.total;
        info.total_avg._organism_age      /= info.total_avg.total;
        info.total_avg._gathered_food     /= info.total_avg.total;
        info.total_avg._mouth_blocks      /= info.total_avg.total;
        info.total_avg._producer_blocks   /= info.total_avg.total;
        info.total_avg._mover_blocks      /= info.total_avg.total;
        info.total_avg._killer_blocks     /= info.total_avg.total;
        info.total_avg._armor_blocks      /= info.total_avg.total;
        info.total_avg._eye_blocks        /= info.total_avg.total;
        info.total_avg.occ_instructions_num = info.total_avg.total_occ_instructions_num > 0 ? info.total_avg.total_occ_instructions_num / info.total_avg.total : 0;

        info.total_avg.brain_mutation_rate   /= info.total_avg.total;
        info.total_avg.anatomy_mutation_rate /= info.total_avg.total;

        if (std::isnan(info.total_avg.size))             {info.total_avg.size = 0;}
        if (std::isnan(info.move_range))                 {info.move_range     = 0;}

        if (std::isnan(info.total_avg._organism_lifetime)) {info.total_avg._organism_lifetime = 0;}
        if (std::isnan(info.total_avg._organism_age))      {info.total_avg._organism_age      = 0;}
        if (std::isnan(info.total_avg._gathered_food))     {info.total_avg._gathered_food     = 0;}
        if (std::isnan(info.total_avg._mouth_blocks))      {info.total_avg._mouth_blocks      = 0;}
        if (std::isnan(info.total_avg._producer_blocks))   {info.total_avg._producer_blocks   = 0;}
        if (std::isnan(info.total_avg._mover_blocks))      {info.total_avg._mover_blocks      = 0;}
        if (std::isnan(info.total_avg._killer_blocks))     {info.total_avg._killer_blocks     = 0;}
        if (std::isnan(info.total_avg._armor_blocks))      {info.total_avg._armor_blocks      = 0;}
        if (std::isnan(info.total_avg._eye_blocks))        {info.total_avg._eye_blocks        = 0;}
        if (std::isnan(info.total_avg.occ_instructions_num)) {info.total_avg.occ_instructions_num = 0;}

        if (std::isnan(info.total_avg.brain_mutation_rate))   {info.total_avg.brain_mutation_rate   = 0;}
        if (std::isnan(info.total_avg.anatomy_mutation_rate)) {info.total_avg.anatomy_mutation_rate = 0;}


        info.moving_avg.size /= info.moving_avg.total;

        info.moving_avg._organism_lifetime /= info.moving_avg.total;
        info.moving_avg._organism_age      /= info.moving_avg.total;
        info.moving_avg._gathered_food     /= info.moving_avg.total;
        info.moving_avg._mouth_blocks      /= info.moving_avg.total;
        info.moving_avg._producer_blocks   /= info.moving_avg.total;
        info.moving_avg._mover_blocks      /= info.moving_avg.total;
        info.moving_avg._killer_blocks     /= info.moving_avg.total;
        info.moving_avg._armor_blocks      /= info.moving_avg.total;
        info.moving_avg._eye_blocks        /= info.moving_avg.total;
        info.moving_avg.occ_instructions_num = info.moving_avg.total_occ_instructions_num > 0 ? info.moving_avg.total_occ_instructions_num / info.moving_avg.total : 0;

        info.moving_avg.brain_mutation_rate   /= info.moving_avg.total;
        info.moving_avg.anatomy_mutation_rate /= info.moving_avg.total;

        if (std::isnan(info.moving_avg.size))             {info.moving_avg.size             = 0;}

        if (std::isnan(info.moving_avg._organism_lifetime)) {info.moving_avg._organism_lifetime = 0;}
        if (std::isnan(info.moving_avg._organism_age))      {info.moving_avg._organism_age      = 0;}
        if (std::isnan(info.moving_avg._gathered_food))     {info.moving_avg._gathered_food     = 0;}
        if (std::isnan(info.moving_avg._mouth_blocks))      {info.moving_avg._mouth_blocks      = 0;}
        if (std::isnan(info.moving_avg._producer_blocks))   {info.moving_avg._producer_blocks   = 0;}
        if (std::isnan(info.moving_avg._mover_blocks))      {info.moving_avg._mover_blocks      = 0;}
        if (std::isnan(info.moving_avg._killer_blocks))     {info.moving_avg._killer_blocks     = 0;}
        if (std::isnan(info.moving_avg._armor_blocks))      {info.moving_avg._armor_blocks      = 0;}
        if (std::isnan(info.moving_avg._eye_blocks))        {info.moving_avg._eye_blocks        = 0;}
        if (std::isnan(info.moving_avg.occ_instructions_num)) {info.moving_avg.occ_instructions_num = 0;}

        if (std::isnan(info.moving_avg.brain_mutation_rate))   {info.moving_avg.brain_mutation_rate   = 0;}
        if (std::isnan(info.moving_avg.anatomy_mutation_rate)) {info.moving_avg.anatomy_mutation_rate = 0;}


        info.station_avg.size /= info.station_avg.total;

        info.station_avg._organism_lifetime /= info.station_avg.total;
        info.station_avg._organism_age      /= info.station_avg.total;
        info.station_avg._gathered_food     /= info.station_avg.total;
        info.station_avg._mouth_blocks      /= info.station_avg.total;
        info.station_avg._producer_blocks   /= info.station_avg.total;
        info.station_avg._mover_blocks      /= info.station_avg.total;
        info.station_avg._killer_blocks     /= info.station_avg.total;
        info.station_avg._armor_blocks      /= info.station_avg.total;
        info.station_avg._eye_blocks        /= info.station_avg.total;
        info.station_avg.occ_instructions_num = info.station_avg.total_occ_instructions_num ? info.station_avg.total_occ_instructions_num / info.station_avg.total : 0;

        info.station_avg.brain_mutation_rate   /= info.station_avg.total;
        info.station_avg.anatomy_mutation_rate /= info.station_avg.total;

        if (std::isnan(info.station_avg.size))             {info.station_avg.size             = 0;}

        if (std::isnan(info.station_avg._organism_lifetime)) {info.station_avg._organism_lifetime = 0;}
        if (std::isnan(info.station_avg._organism_age))      {info.station_avg._organism_age      = 0;}
        if (std::isnan(info.station_avg._gathered_food))     {info.station_avg._gathered_food     = 0;}
        if (std::isnan(info.station_avg._mouth_blocks))      {info.station_avg._mouth_blocks      = 0;}
        if (std::isnan(info.station_avg._producer_blocks))   {info.station_avg._producer_blocks   = 0;}
        if (std::isnan(info.station_avg._mover_blocks))      {info.station_avg._mover_blocks      = 0;}
        if (std::isnan(info.station_avg._killer_blocks))     {info.station_avg._killer_blocks     = 0;}
        if (std::isnan(info.station_avg._armor_blocks))      {info.station_avg._armor_blocks      = 0;}
        if (std::isnan(info.station_avg._eye_blocks))        {info.station_avg._eye_blocks        = 0;}
        if (std::isnan(info.station_avg.occ_instructions_num)) {info.station_avg.occ_instructions_num = 0;}

        if (std::isnan(info.station_avg.brain_mutation_rate))   {info.station_avg.brain_mutation_rate   = 0;}
        if (std::isnan(info.station_avg.anatomy_mutation_rate)) {info.station_avg.anatomy_mutation_rate = 0;}
    }
};
#endif //LIFEENGINEEXTENDED_ORGANISMINFOCONTAINER_H
