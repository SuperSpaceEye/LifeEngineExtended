//
// Created by spaceeye on 26.07.22.
//

#include "Recorder.h"

Recorder::Recorder(Ui::MainWindow *_parent_ui, EngineDataContainer * edc, EngineControlParameters * ecp): parent_ui(_parent_ui), edc(edc), ecp(ecp) {
    _ui.setupUi(this);
}

void Recorder::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_recorder_window->setChecked(false);
    QWidget::closeEvent(event);
}

OrganismAvgBlockInformation Recorder::parse_organisms_info() {
    OrganismAvgBlockInformation info;

    bool has_pool = true;
    int i = 0;
    //Why while loop? the easier implementation with for loop randomly crashes sometimes, and I don't know why.
    while (has_pool) {
        std::vector<Organism*> * pool;

        if (ecp->simulation_mode == SimulationModes::CPU_Single_Threaded) {
            pool = &edc->organisms;
            has_pool = false;
        } else if (ecp->simulation_mode == SimulationModes::CPU_Partial_Multi_threaded) {
            pool = &edc->organisms_pools[i];
            i++;
            if (i >= ecp->num_threads) {
                has_pool = false;
            }
        } else {
            throw "no pool";
        }

        for (auto & organism: *pool) {
            info.total_size_organism_blocks += organism->anatomy->_organism_blocks.size();
            info.total_size_producing_space += organism->anatomy->_producing_space.size();
            info.total_size_eating_space    += organism->anatomy->_eating_space.size();

            if (organism->anatomy->_mover_blocks > 0) {
                info.move_range += organism->move_range;
                info.moving_organisms++;

                if (organism->anatomy->_eye_blocks > 0) {
                    info.organisms_with_eyes++;
                }
            }

            info.total_avg.size += organism->anatomy->_organism_blocks.size();

            info.total_avg._organism_lifetime += organism->max_lifetime;
            info.total_avg._organism_age      += organism->lifetime;
            info.total_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
            info.total_avg._producer_blocks   += organism->anatomy->_producer_blocks;
            info.total_avg._mover_blocks      += organism->anatomy->_mover_blocks;
            info.total_avg._killer_blocks     += organism->anatomy->_killer_blocks;
            info.total_avg._armor_blocks      += organism->anatomy->_armor_blocks;
            info.total_avg._eye_blocks        += organism->anatomy->_eye_blocks;

            info.total_avg.brain_mutation_rate   += organism->brain_mutation_rate;
            info.total_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
            info.total_avg.total++;

            if (organism->anatomy->_mover_blocks > 0) {
                info.moving_avg.size += organism->anatomy->_organism_blocks.size();

                info.moving_avg._organism_lifetime += organism->max_lifetime;
                info.moving_avg._organism_age      += organism->lifetime;
                info.moving_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
                info.moving_avg._producer_blocks   += organism->anatomy->_producer_blocks;
                info.moving_avg._mover_blocks      += organism->anatomy->_mover_blocks;
                info.moving_avg._killer_blocks     += organism->anatomy->_killer_blocks;
                info.moving_avg._armor_blocks      += organism->anatomy->_armor_blocks;
                info.moving_avg._eye_blocks        += organism->anatomy->_eye_blocks;

                info.moving_avg.brain_mutation_rate   += organism->brain_mutation_rate;
                info.moving_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
                info.moving_avg.total++;
            } else {
                info.station_avg.size += organism->anatomy->_organism_blocks.size();

                info.station_avg._organism_lifetime += organism->max_lifetime;
                info.station_avg._organism_age      += organism->lifetime;
                info.station_avg._mouth_blocks      += organism->anatomy->_mouth_blocks;
                info.station_avg._producer_blocks   += organism->anatomy->_producer_blocks;
                info.station_avg._mover_blocks      += organism->anatomy->_mover_blocks;
                info.station_avg._killer_blocks     += organism->anatomy->_killer_blocks;
                info.station_avg._armor_blocks      += organism->anatomy->_armor_blocks;
                info.station_avg._eye_blocks        += organism->anatomy->_eye_blocks;

                info.station_avg.brain_mutation_rate   += organism->brain_mutation_rate;
                info.station_avg.anatomy_mutation_rate += organism->anatomy_mutation_rate;
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
                      (sizeof(Organism) * info.total_avg.total)
            ;

    info.total_total_mutation_rate = info.total_avg.anatomy_mutation_rate;

    info.total_avg.size /= info.total_avg.total;

    info.total_avg._organism_lifetime /= info.total_avg.total;
    info.total_avg._organism_age      /= info.total_avg.total;
    info.total_avg._mouth_blocks      /= info.total_avg.total;
    info.total_avg._producer_blocks   /= info.total_avg.total;
    info.total_avg._mover_blocks      /= info.total_avg.total;
    info.total_avg._killer_blocks     /= info.total_avg.total;
    info.total_avg._armor_blocks      /= info.total_avg.total;
    info.total_avg._eye_blocks        /= info.total_avg.total;

    info.total_avg.brain_mutation_rate   /= info.total_avg.total;
    info.total_avg.anatomy_mutation_rate /= info.total_avg.total;

    if (std::isnan(info.total_avg.size))             {info.total_avg.size = 0;}
    if (std::isnan(info.move_range))                 {info.move_range     = 0;}

    if (std::isnan(info.total_avg._organism_lifetime)) {info.total_avg._organism_lifetime = 0;}
    if (std::isnan(info.total_avg._organism_age))      {info.total_avg._organism_age      = 0;}
    if (std::isnan(info.total_avg._mouth_blocks))      {info.total_avg._mouth_blocks      = 0;}
    if (std::isnan(info.total_avg._producer_blocks))   {info.total_avg._producer_blocks   = 0;}
    if (std::isnan(info.total_avg._mover_blocks))      {info.total_avg._mover_blocks      = 0;}
    if (std::isnan(info.total_avg._killer_blocks))     {info.total_avg._killer_blocks     = 0;}
    if (std::isnan(info.total_avg._armor_blocks))      {info.total_avg._armor_blocks      = 0;}
    if (std::isnan(info.total_avg._eye_blocks))        {info.total_avg._eye_blocks        = 0;}

    if (std::isnan(info.total_avg.brain_mutation_rate))   {info.total_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.total_avg.anatomy_mutation_rate)) {info.total_avg.anatomy_mutation_rate = 0;}


    info.moving_avg.size /= info.moving_avg.total;

    info.moving_avg._organism_lifetime /= info.moving_avg.total;
    info.moving_avg._organism_age      /= info.moving_avg.total;
    info.moving_avg._mouth_blocks      /= info.moving_avg.total;
    info.moving_avg._producer_blocks   /= info.moving_avg.total;
    info.moving_avg._mover_blocks      /= info.moving_avg.total;
    info.moving_avg._killer_blocks     /= info.moving_avg.total;
    info.moving_avg._armor_blocks      /= info.moving_avg.total;
    info.moving_avg._eye_blocks        /= info.moving_avg.total;

    info.moving_avg.brain_mutation_rate   /= info.moving_avg.total;
    info.moving_avg.anatomy_mutation_rate /= info.moving_avg.total;

    if (std::isnan(info.moving_avg.size))             {info.moving_avg.size             = 0;}

    if (std::isnan(info.moving_avg._organism_lifetime)) {info.moving_avg._organism_lifetime = 0;}
    if (std::isnan(info.moving_avg._organism_age))      {info.moving_avg._organism_age      = 0;}
    if (std::isnan(info.moving_avg._mouth_blocks))      {info.moving_avg._mouth_blocks      = 0;}
    if (std::isnan(info.moving_avg._producer_blocks))   {info.moving_avg._producer_blocks   = 0;}
    if (std::isnan(info.moving_avg._mover_blocks))      {info.moving_avg._mover_blocks      = 0;}
    if (std::isnan(info.moving_avg._killer_blocks))     {info.moving_avg._killer_blocks     = 0;}
    if (std::isnan(info.moving_avg._armor_blocks))      {info.moving_avg._armor_blocks      = 0;}
    if (std::isnan(info.moving_avg._eye_blocks))        {info.moving_avg._eye_blocks        = 0;}

    if (std::isnan(info.moving_avg.brain_mutation_rate))   {info.moving_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.moving_avg.anatomy_mutation_rate)) {info.moving_avg.anatomy_mutation_rate = 0;}


    info.station_avg.size /= info.station_avg.total;

    info.station_avg._organism_lifetime /= info.station_avg.total;
    info.station_avg._organism_age      /= info.station_avg.total;
    info.station_avg._mouth_blocks      /= info.station_avg.total;
    info.station_avg._producer_blocks   /= info.station_avg.total;
    info.station_avg._mover_blocks      /= info.station_avg.total;
    info.station_avg._killer_blocks     /= info.station_avg.total;
    info.station_avg._armor_blocks      /= info.station_avg.total;
    info.station_avg._eye_blocks        /= info.station_avg.total;

    info.station_avg.brain_mutation_rate   /= info.station_avg.total;
    info.station_avg.anatomy_mutation_rate /= info.station_avg.total;

    if (std::isnan(info.station_avg.size))             {info.station_avg.size             = 0;}

    if (std::isnan(info.station_avg._organism_lifetime)) {info.station_avg._organism_lifetime = 0;}
    if (std::isnan(info.station_avg._organism_age))      {info.station_avg._organism_age      = 0;}
    if (std::isnan(info.station_avg._mouth_blocks))      {info.station_avg._mouth_blocks      = 0;}
    if (std::isnan(info.station_avg._producer_blocks))   {info.station_avg._producer_blocks   = 0;}
    if (std::isnan(info.station_avg._mover_blocks))      {info.station_avg._mover_blocks      = 0;}
    if (std::isnan(info.station_avg._killer_blocks))     {info.station_avg._killer_blocks     = 0;}
    if (std::isnan(info.station_avg._armor_blocks))      {info.station_avg._armor_blocks      = 0;}
    if (std::isnan(info.station_avg._eye_blocks))        {info.station_avg._eye_blocks        = 0;}

    if (std::isnan(info.station_avg.brain_mutation_rate))   {info.station_avg.brain_mutation_rate   = 0;}
    if (std::isnan(info.station_avg.anatomy_mutation_rate)) {info.station_avg.anatomy_mutation_rate = 0;}
    return info;
}