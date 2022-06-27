//
// Created by spaceeye on 27.06.22.
//

#ifndef THELIFEENGINECPP_CURSORMODE_H
#define THELIFEENGINECPP_CURSORMODE_H

enum class CursorMode {
    ModifyFood,
    ModifyWall,
    KillOrganism,
    ChooseOrganism,
    PlaceOrganism,
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
#include "pix_pos.h"
#include "textures.h"
#include "../MainWindow/WindowUI.h"
#include "../Statistics/StatisticsCore.h"
#include "cuda_image_creator.cuh"
#include "get_device_count.cuh"

#endif //THELIFEENGINECPP_CURSORMODE_H
