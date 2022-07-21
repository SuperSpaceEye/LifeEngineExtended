//
// Created by spaceeye on 21.07.22.
//

#ifndef LIFEENGINEEXTENDED_ORGANISMDATA_H
#define LIFEENGINEEXTENDED_ORGANISMDATA_H

struct OrganismData {
    int x = 0;
    int y = 0;
    int max_lifetime = 0;
    int lifetime = 0;
    int move_range = 1;
    int move_counter = 0;
    int max_decision_lifetime = 2;
    int max_do_nothing_lifetime = 4;
    float anatomy_mutation_rate = 0.05;
    float brain_mutation_rate = 0.1;
    float food_collected = 0;
    float food_needed = 0;
    float life_points = 0;
    float damage = 0;
    Rotation rotation = Rotation::UP;
    DecisionObservation last_decision = DecisionObservation{};
};

#include <iostream>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <iomanip>
#include <thread>
#include <random>
#include <fstream>
#include <filesystem>
#include <boost/lexical_cast.hpp>
#include <boost/lexical_cast/try_lexical_convert.hpp>
#include <boost/nondet_random.hpp>
#include <boost/random.hpp>
#include <boost/property_tree/ptree.hpp>
#include <QApplication>
#include <QWidget>
#include <QTimer>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QMessageBox>
#include <QLineEdit>
#include <QDialog>
#include <QFont>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QToolBar>
#include <QWheelEvent>
#include "../CustomJsonParser/json_parser.hpp"
#include "../SimulationEngine/SimulationEngine.h"
#include "../Containers/CPU/ColorContainer.h"
#include "../Containers/CPU/SimulationParameters.h"
#include "../Containers/CPU/EngineControlContainer.h"
#include "../Containers/CPU/EngineDataContainer.h"
#include "../Containers/CPU/OrganismBlockParameters.h"
#include "../OrganismEditor/OrganismEditor.h"
#include "../PRNGS/lehmer64.h"
#include "textures.h"
#include "MiscFuncs.h"
#include "CursorMode.h"
#include "Vector2.h"
#include "../MainWindow/WindowUI.h"
#include "../Statistics/StatisticsCore.h"
#include "cuda_image_creator.cuh"
#include "get_device_count.cuh"

#endif //LIFEENGINEEXTENDED_ORGANISMDATA_H
