//
// Created by spaceeye on 05.06.22.
//

#ifndef THELIFEENGINECPP_TEXTURES_H
#define THELIFEENGINECPP_TEXTURES_H

#include <vector>

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

struct Textures {
    color rawEyeTexture[5*5] = {GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                GRAY1, GRAY2, BLACK1, GRAY2, GRAY1,
                                GRAY1, GRAY3, BLACK2, GRAY3, GRAY1,
                                GRAY1, GRAY1, GRAY1,  GRAY1, GRAY1};
};

#endif //THELIFEENGINECPP_TEXTURES_H
