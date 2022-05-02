//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_UIWINDOW_H
#define LANGUAGES_UIWINDOW_H

#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <omp.h>
#include <random>
#include <boost/lexical_cast.hpp>
#include <boost/lexical_cast/try_lexical_convert.hpp>

#include <QApplication>
#include <QWidget>
#include <QTimer>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QMessageBox>

#include "SimulationEngine.h"
#include "ColorContainer.h"
#include "ParametersStruct.h"
#include "EngineControlContainer.h"
#include "EngineDataContainer.h"
#include "OrganismBlockParameters.h"
#include "WindowUI.h"

enum class CursorMode {
    Food_mode,
    Wall_mode,
    Kill_mode,
};

class WindowCore: public QWidget {
        Q_OBJECT
private:
    int window_width;
    int window_height;

    // by how much pixels is allowed to be after the end of simulation image.
    // (blank space after dragging image)
    int allow_num_padding = 50;

    // relative_x>1
    float scaling_coefficient = 1.05;
    float scaling_zoom = 1;

    bool right_mouse_button_pressed = false;
    bool left_mouse_button_pressed = false;

    float center_x = window_width/2;
    float center_y = window_height/2;

    CursorMode cursor_mode = CursorMode::Food_mode;

    int start_height = 100;

    void move_center(int delta_x, int delta_y);
    void reset_image();

    int last_mouse_x = 0;
    int last_mouse_y = 0;

    SimulationEngine* engine;

    float window_interval = 0.;
    long delta_window_processing_time = 0;
    bool allow_waiting_overhead = false;

    Ui::MainWindow _ui;
    QTimer * timer;
    //void showEvent(QShowEvent *event);
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_window_update;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_simulation_update;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;
    int window_frames = 0;
    int simulation_frames = 0;

    ColorContainer color_container;

    ParametersStruct parameters{};

    EngineControlParameters cp;
    EngineDataContainer dc;
    OrganismBlockParameters op;

    std::thread engine_thread;
    std::mutex engine_mutex;

    QImage image;
    std::vector<unsigned char> image_vector;
    QGraphicsScene scene;
    QGraphicsPixmapItem pixmap_item;

    bool pause_image_construction = false;
    bool full_simulation_grid_parsed = false;

    bool stop_console_output = false;
    bool synchronise_simulation_and_window = false;

    void mainloop_tick();
    void window_tick();
    void set_simulation_interval(int max_simulation_fps);
    void set_window_interval(int max_window_fps);
    void update_fps_labels(int fps, int sps);
    void resize_image();
    void set_image_pixel(int x, int y, QColor & color);
    bool compare_pixel_color(int x, int y, QColor & color);

    void create_image();

    QColor inline &get_color(BlockTypes type);

    bool wait_for_engine_to_pause();
    void parse_simulation_grid(std::vector<int> & lin_width, std::vector<int> & lin_height);
    void parse_full_simulation_grid(bool parse);

    void set_simulation_num_threads(uint8_t num_threads);

    void set_cursor_mode(CursorMode mode);
    void set_simulation_mode(SimulationModes mode);

    void inline calculate_linspace(std::vector<int> & lin_width, std::vector<int> & lin_height,
                            int start_x,  int end_x, int start_y, int end_y, int image_width, int image_height);
    void inline calculate_truncated_linspace(int image_width, int image_height,
                                      std::vector<int> & lin_width,
                                      std::vector<int> & lin_height,
                                      std::vector<int> & truncated_lin_width,
                                      std::vector<int> & truncated_lin_height);
    void inline image_for_loop(int image_width, int image_height,
                               std::vector<int> & lin_width,
                               std::vector<int> & lin_height);

    void closeEvent(QCloseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    bool eventFilter(QObject *watched, QEvent *event);

private slots:
    void pause_slot(bool paused);
    void stoprender_slot(bool stopped_render);
    void clear_slot();
    void reset_slot();
    void pass_one_tick_slot();
    void reset_view_slot();

    void parse_max_sps_slot();
    void parse_max_fps_slot();
    void parse_num_threads_slot();

    void food_rbutton_slot();
    void wall_rbutton_slot();
    void kill_rbutton_slot();

    void single_thread_rbutton_slot();
    void multi_thread_rbutton_slot();
    void cuda_rbutton_slot();

    void stop_console_output_slot(bool state);
    void synchronise_simulation_and_window_slot(bool state);

public:
    WindowCore(int window_width, int window_height,
               int simulation_width, int simulation_height,
               int window_fps, int simulation_fps, int simulation_num_threads,
               QWidget *parent);
};


#endif //LANGUAGES_UIWINDOW_H
