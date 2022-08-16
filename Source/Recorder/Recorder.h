//
// Created by spaceeye on 26.07.22.
//

#ifndef LIFEENGINEEXTENDED_RECORDER_H
#define LIFEENGINEEXTENDED_RECORDER_H

#include <iostream>
#include <filesystem>
#include <ctime>

#include <QFileDialog>
#include <QImage>

#include "RecorderWindowUI.h"
#include "../MainWindow/WindowUI.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Stuff/MiscFuncs.h"
#include "../Containers/CPU/EngineControlParametersContainer.h"
#include "../Organism/CPU/Organism.h"
#include "../Organism/CPU/Brain.h"
#include "../Organism/CPU/Anatomy.h"
#include "../SimulationEngine/SimulationEngine.h"
#include "../Stuff/textures.h"
#include "../Stuff/ImageCreation.h"
#include "../Containers/CPU/OrganismInfoContainer.h"

#if defined(__WIN32)
#include <windows.h>
#include <cwchar>
#endif

class Recorder: public QWidget {
    Q_OBJECT
private:
    Ui::Recorder _ui{};
    Ui::MainWindow * parent_ui = nullptr;
    EngineDataContainer * edc = nullptr;
    EngineControlParameters * ecp = nullptr;
    SimulationEngine * engine = nullptr;
    ColorContainer * cc = nullptr;
    TexturesContainer * textures = nullptr;
    RecordingData * recd;

    std::string ffmpeg_path = "ffmpeg";

    int num_pixels_per_block = 5;
    bool recording_paused = false;
    int video_fps = 60;

    void closeEvent(QCloseEvent * event) override;

    void create_image(std::vector<unsigned char> &raw_image_data, std::vector<BaseGridBlock> &grid,
                      int simulation_width, int simulation_height, int num_pixels_per_block);

    void init_gui();

    std::string new_recording(std::string path);

    static std::string get_string_date();

    void clear_data();

public:
    Recorder(Ui::MainWindow * _parent_ui, EngineDataContainer * edc, EngineControlParameters * ecp, ColorContainer * cc, TexturesContainer * textures,
             RecordingData * recording_data);

    void set_engine(SimulationEngine * engine);

    void update_label();
private slots:
    void le_number_of_pixels_per_block_slot();
    void le_first_grid_buffer_size_slot();
    void le_log_every_n_tick_slot();
    void le_video_fps_slot();

    void b_create_image_slot();
    void b_start_recording_slot();
    void b_stop_recording_slot();
    void b_load_intermediate_data_location_slot();
    void b_compile_intermediate_data_into_video_slot();
    void b_clear_intermediate_data_slot();
    void b_delete_all_intermediate_data_from_disk_slot();
    void b_new_recording_slot();
    void b_pause_recording_slot();
};

#endif //LIFEENGINEEXTENDED_RECORDER_H
