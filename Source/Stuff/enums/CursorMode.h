// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 27.06.22.
//

#ifndef THELIFEENGINECPP_CURSORMODE_H
#define THELIFEENGINECPP_CURSORMODE_H

enum class CursorMode {
    NoAction,
    ModifyFood,
    ModifyWall,
    KillOrganism,
    ChooseOrganism,
    PlaceOrganism,

    DebugDisplayInfo,
};

#endif //THELIFEENGINECPP_CURSORMODE_H
