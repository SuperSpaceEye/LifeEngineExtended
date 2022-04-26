//
// Created by spaceeye on 16.03.2022.
//

#ifndef LANGUAGES_UIWINDOW_H
#define LANGUAGES_UIWINDOW_H

#include <iostream>
#include <TGUI/TGUI.hpp>
#include <chrono>
#include <cmath>
#include <thread>
#include <omp.h>
#include <random>
#include "SimulationEngine.h"
#include "ColorContainer.h"
#include "UIElemets/GridLayout2D.h"
#include "ParametersStruct.h"
#include "EngineControlContainer.h"
#include "EngineDataContainer.h"
#include "OrganismBlockParameters.h"

class UIWindow {
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

    //TODO why do i need this again?
    int image_width;
    int image_height;

    int start_height = 100;

    void make_image();
    void move_center(int delta_x, int delta_y);
    void reset_image();

    int last_mouse_x = 0;
    int last_mouse_y = 0;

    SimulationEngine* engine;

    sf::Image image;
    sf::Color base_simulation_color = sf::Color::White;

    sf::Texture menu_texture;
    sf::Image menu_image;
    sf::Sprite menu_sprite;
    tgui::CanvasSFML::Ptr menu_canvas;

    sf::RenderWindow window;
    sf::Font font;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::ContextSettings settings;

    tgui::GuiSFML gui;
    tgui::CanvasSFML::Ptr canvas;
    std::vector<GridLayout2D*> layouts;

    // separate from each other
    // if <=0 then unlimited
    int max_window_fps;
    int max_simulation_fps;

    float window_interval = 0.;
    long delta_window_processing_time = 0;
    bool allow_waiting_overhead = false;

    //percentage for menu
    float menu_size = 0.33;
    bool menu_shows = true;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_window_update;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_simulation_update;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_timer;
    int window_frames = 0;
    int simulation_frames = 0;

    ColorContainer color_container;

    tgui::Label::Ptr window_fps_label;
    tgui::Label::Ptr simulation_fps_label;

    std::vector<tgui::Widget::Ptr> menu_widgets;

    ParametersStruct parameters{};

    EngineControlParameters cp;
    EngineDataContainer dc;
    OrganismBlockParameters op;

    std::thread engine_thread;
    std::mutex engine_mutex;

    bool pause_image_construction = false;
    bool full_simulation_grid_parsed = false;

    static std::vector<double> linspace(double start, double end, int num);

    void window_tick();
    void set_simulation_interval();
    void set_window_interval();
    void make_visuals();
    void update_fps_labels(int fps, int sps);

    void configure_simulation_canvas();
    void configure_menu_canvas();

    void create_image();

    inline sf::Color& get_color(BlockTypes type);

    bool wait_for_engine_to_pause();
    void parse_simulation_grid(std::vector<int> lin_width, std::vector<int> lin_height);
    void parse_full_simulation_grid(bool parse);

    void set_simulation_num_threads(uint8_t num_threads);

public:
    UIWindow(int window_width, int window_height,
             int simulation_width, int simulation_height,
             int window_fps, int simulation_fps, int simulation_num_threads);

    // the main loop of the program
    void main_loop();

};


#endif //LANGUAGES_UIWINDOW_H
