//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include "../../textures.h"

struct ColorContainer {
    //QColor menu_color = QColor{200, 200, 255};
    color simulation_background_color = color{58, 75, 104};
    color organism_boundary = color{255, 0, 0};

    color empty_block = color{14, 19, 24};
    color mouth =       color{222,177,77};
    color producer =    color{21, 222,89};
    color mover =       color{96, 212,255};
    color killer =      color{248,35, 128};
    color armor =       color{114,48, 219};
    color eye =         color{182,193,234};
    color food =        color{47, 122,183};
    color wall =        color{128, 128, 128};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
