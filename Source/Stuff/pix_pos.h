// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 05.06.22.
//

#ifndef THELIFEENGINECPP_PIX_POS_H
#define THELIFEENGINECPP_PIX_POS_H

struct pix_pos {
    int start = 0;
    int stop = 0;
    pix_pos(int start, int stop): start(start), stop(stop) {}
};

#endif //THELIFEENGINECPP_PIX_POS_H
