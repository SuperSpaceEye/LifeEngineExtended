// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 05.06.22.
//

#ifndef THELIFEENGINECPP_TEXTURES_H
#define THELIFEENGINECPP_TEXTURES_H

#include <vector>
#include <array>

struct color {
    unsigned char r{0};
    unsigned char g{0};
    unsigned char b{0};
};

#define BLACK1 color{14,  19,  24}
#define BLACK2 color{56,  62,  77}
#define GRAY1  color{182, 193, 234}
#define GRAY2  color{161, 172, 209}
#define GRAY3  color{167, 177, 215}

struct TextureHolder {
    int width;
    int height;
    std::vector<color> texture;
};

const std::array<TextureHolder, 9> default_holders {
        TextureHolder{1, 1, std::vector<color>{color{14, 19,  24}}}, //EmptyBlock
        TextureHolder{1, 1, std::vector<color>{color{222,177, 77}}}, //MouthBlock
        TextureHolder{1, 1, std::vector<color>{color{21, 222, 89}}}, //ProducerBlock
        TextureHolder{1, 1, std::vector<color>{color{96, 212,255}}}, //MoverBlock
        TextureHolder{1, 1, std::vector<color>{color{248,35, 128}}}, //KillerBlock
        TextureHolder{1, 1, std::vector<color>{color{114,48, 219}}}, //ArmorBlock
        TextureHolder{5, 5, std::vector<color>{GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                               GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                               GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                               GRAY1, GRAY3, BLACK2, GRAY3, GRAY1,
                                               GRAY1, GRAY1, GRAY1,  GRAY1, GRAY1}}, //EyeBlock
        TextureHolder{1, 1, std::vector<color>{color{47, 122,183}}}, //FoodBlock
        TextureHolder{1, 1, std::vector<color>{color{128,128,128}}}, //WallBlock
};

struct TexturesContainer {
    std::array<TextureHolder, 9> textures {default_holders};
};

#endif //THELIFEENGINECPP_TEXTURES_H
