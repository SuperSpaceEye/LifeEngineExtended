// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include "Stuff/ImageStuff/textures.h"

struct ColorContainer {
    //QColor menu_color = QColor{200, 200, 255};
    Textures::color simulation_background_color {58, 75, 104};
    Textures::color organism_boundary {255, 0, 0};

    Textures::color empty_block {14, 19, 24};
    Textures::color mouth       {222, 177, 77};
    Textures::color producer    {21, 222, 89};
    Textures::color mover       {96, 212, 255};
    Textures::color killer      {248, 35, 128};
    Textures::color armor       {114, 48, 219};
    Textures::color eye         {182, 193, 234};
    Textures::color food        {47, 122, 183};
    Textures::color wall        {128, 128, 128};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
