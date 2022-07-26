//
// Created by spaceeye on 26.07.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDER_H
#define LIFEENGINEEXTENDED_RECORDER_H

#include <iostream>
#ifndef __WIN32
#include <filesystem>
#endif

#include <QFileDialog>
#include <QImage>

#include "RecorderWindowUI.h"
#include "../MainWindow/WindowUI.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Stuff/MiscFuncs.h"
#include "../Containers/CPU/EngineControlContainer.h"
#include "../Organism/CPU/Organism.h"
#include "../Organism/CPU/Brain.h"
#include "../Organism/CPU/Anatomy.h"
#include "../SimulationEngine/SimulationEngine.h"
#include "../Stuff/textures.h"

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
    SimulationEngine * engine = nullptr;
    ColorContainer * cc = nullptr;
    Textures * textures = nullptr;

    int num_pixels_per_block = 5;

    void closeEvent(QCloseEvent * event) override;

    void create_image(std::vector<unsigned char> &raw_image_data);

    void complex_image_creation(const std::vector<int> &lin_width, const std::vector<int> &lin_height,
                                std::vector<unsigned char> &raw_image_vector);
    void set_image_pixel(int x, int y, const color &color, std::vector<unsigned char> &image_vector);
    color & get_texture_color(BlockTypes type, Rotation rotation, float relative_x_scale, float relative_y_scale);

    void init_gui();
public:
    Recorder(Ui::MainWindow * _parent_ui, EngineDataContainer * edc, EngineControlParameters * ecp, ColorContainer * cc, Textures * textures);

    OrganismAvgBlockInformation parse_organisms_info();

    void set_engine(SimulationEngine * engine);

private slots:
    void le_number_of_pixels_per_block_slot();

    void b_create_image_slot();
};

#endif //LIFEENGINEEXTENDED_RECORDER_H
