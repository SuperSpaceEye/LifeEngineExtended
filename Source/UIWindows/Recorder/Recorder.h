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
#include "../../Containers/CPU/EngineDataContainer.h"
#include "../../Stuff/MiscFuncs.h"
#include "../../Containers/CPU/EngineControlParametersContainer.h"
#include "../../Organism/CPU/Organism.h"
#include "../../Organism/CPU/Brain.h"
#include "../../Organism/CPU/Anatomy.h"
#include "../../SimulationEngine/SimulationEngine.h"
#include "../../Stuff/textures.h"
#include "../../Stuff/ImageCreation.h"
#include "../../Containers/CPU/OrganismInfoContainer.h"
#include "../../WorldRecorder/RecordingReconstructor.h"

#ifdef __CUDA_USED__
#include "../../WorldRecorder/RecordingReconstructorCUDA.cuh"
#endif

#include "../../Stuff/moviemaker/include/movie.h"

#if defined(__WIN32)
#include <windows.h>
#include <cwchar>
#endif

class Recorder: public QWidget {
    Q_OBJECT
private:
    Ui::Recorder ui{};
    Ui::MainWindow * parent_ui = nullptr;
    EngineDataContainer * edc = nullptr;
    EngineControlParameters * ecp = nullptr;
    SimulationEngine * engine = nullptr;
    ColorContainer * cc = nullptr;
    TexturesContainer * textures = nullptr;
    TransactionBuffer * tbuffer = nullptr;

    RecordingReconstructor reconstructor;

#ifdef __CUDA_USED__
//    RecordingReconstructorCUDA cuda_reconstructor{};
#endif

    int num_pixels_per_block = 5;
    bool recording_paused = false;
    int video_fps = 60;
    int buffer_size = 5000;

    const float * main_viewpoint_x;
    const float * main_viewpoint_y;
    const float * main_zoom;

    float viewpoint_x = 0;
    float viewpoint_y = 0;
    float zoom = 0;

    int image_width = 1000;
    int image_height = 1000;

    bool use_viewpoint = false;
    bool use_cuda = false;
    bool & cuda_is_available;
    bool compiling_recording = false;

    bool use_cuda_reconstructor = false;

    void closeEvent(QCloseEvent * event) override;

    void create_image(std::vector<unsigned char> &raw_image_data, const std::vector<BaseGridBlock> &grid,
                      int simulation_width, int simulation_height, int num_pixels_per_block, bool use_cuda,
                      bool use_viewpoint, bool yuv_format);

    std::string new_recording(std::string path);

    static std::string get_string_date();

    void clear_data();

    void prepare_relative_view(std::vector<int> &lin_height, std::vector<int> &truncated_lin_width,
                               std::vector<int> &truncated_lin_height, int &image_width, int &image_height,
                               int &start_x,
                               int &end_x, int &start_y, int &end_y, std::vector<int> &lin_width) const;

    void
    prepare_full_view(int simulation_width, int simulation_height, int num_pixels_per_block, int &image_height,
                      int start_x,
                      int end_x, int start_y, int end_y, std::vector<int> &truncated_lin_width,
                      std::vector<int> &truncated_lin_height, int &image_width, std::vector<int> &lin_width,
                      std::vector<int> &lin_height) const;


    void start_normal_thread();

public:
    Recorder(Ui::MainWindow *_parent_ui, EngineDataContainer *edc, EngineControlParameters *ecp, ColorContainer *cc,
             TexturesContainer *textures, TransactionBuffer *tbuffer, float *viewpoint_x, float *viewpoint_y,
             float *zoom, bool &cuda_is_available);

    void set_engine(SimulationEngine * engine);

    void update_label();

    void init_gui();

#ifdef __CUDA_USED__
    CUDAImageCreator cuda_image_creator{};
#endif
private slots:
    void le_number_of_pixels_per_block_slot();
    void le_first_grid_buffer_size_slot();
    void le_log_every_n_tick_slot();
    void le_video_fps_slot();
    void le_zoom_slot();
    void le_viewpoint_y_slot();
    void le_viewpoint_x_slot();
    void le_image_width_slot();
    void le_image_height_slot();

    void b_create_image_slot();
    void b_start_recording_slot();
    void b_stop_recording_slot();
    void b_load_intermediate_data_location_slot();
    void b_compile_intermediate_data_into_video_slot();
    void b_clear_intermediate_data_slot();
    void b_delete_all_intermediate_data_from_disk_slot();
    void b_new_recording_slot();
    void b_pause_recording_slot();
    void b_set_from_camera_slot();

    void cb_use_relative_viewpoint_slot(bool state);
    void cb_use_cuda_slot(bool state);
    void cb_use_cuda_reconstructor_slot(bool state);
};

#endif //LIFEENGINEEXTENDED_RECORDER_H
