//
// Created by spaceeye on 26.07.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDER_H
#define LIFEENGINEEXTENDED_RECORDER_H

#include <QFileDialog>

#include "RecorderWindowUI.h"
#include "../MainWindow/WindowUI.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Stuff/MiscFuncs.h"
#include "../Containers/CPU/EngineControlContainer.h"
#include "../Organism/CPU/Organism.h"
#include "../Organism/CPU/Brain.h"
#include "../Organism/CPU/Anatomy.h"

struct OrganismInfoHolder {
    double size = 0;
    double _organism_lifetime = 0;
    double _organism_age    = 0;
    double _mouth_blocks    = 0;
    double _producer_blocks = 0;
    double _mover_blocks    = 0;
    double _killer_blocks   = 0;
    double _armor_blocks    = 0;
    double _eye_blocks      = 0;
    double brain_mutation_rate = 0;
    double anatomy_mutation_rate = 0;
    int total = 0;
};
struct OrganismAvgBlockInformation {
    uint64_t total_size_organism_blocks = 0;
    uint64_t total_size_producing_space = 0;
    uint64_t total_size_eating_space    = 0;
    uint64_t total_size_single_adjacent_space = 0;
    uint64_t total_size_single_diagonal_adjacent_space = 0;
    uint64_t total_size = 0;

    OrganismInfoHolder total_avg{};
    OrganismInfoHolder station_avg{};
    OrganismInfoHolder moving_avg{};

    double move_range = 0;
    int moving_organisms = 0;
    int organisms_with_eyes = 0;

    double total_total_mutation_rate = 0;
};

class Recorder: public QWidget {
    Q_OBJECT
private:
    Ui::Recorder _ui{};
    Ui::MainWindow * parent_ui = nullptr;
    EngineDataContainer * edc = nullptr;
    EngineControlParameters * ecp = nullptr;

    int num_pixels_per_block = 5;

    void closeEvent(QCloseEvent * event) override;
public:
    Recorder(Ui::MainWindow * _parent_ui, EngineDataContainer * edc, EngineControlParameters * ecp);

    OrganismAvgBlockInformation parse_organisms_info();

private slots:
    void le_number_of_pixels_per_block_slot();

    void b_create_image_slot();
};

#endif //LIFEENGINEEXTENDED_RECORDER_H
