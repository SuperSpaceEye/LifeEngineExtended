//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include "../../Stuff/textures.h"

struct ColorContainer {
    //QColor menu_color = QColor{200, 200, 255};
    color simulation_background_color {58, 75, 104};
    color organism_boundary {255, 0, 0};

    color empty_block {14, 19, 24};
    color mouth       {222,177,77};
    color producer    {21, 222,89};
    color mover       {96, 212,255};
    color killer      {248,35, 128};
    color armor       {114,48, 219};
    color eye         {182,193,234};
    color food        {47, 122,183};
    color wall        {128, 128, 128};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
