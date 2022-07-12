// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 22.05.22.
//

#ifndef THELIFEENGINECPP_ACTIONS_H
#define THELIFEENGINECPP_ACTIONS_H

enum class ActionType {
    TryAddFood,
    TryRemoveFood,
    TryAddWall,
    TryRemoveWall,
    TryAddOrganism,
    TryKillOrganism,
    TrySelectOrganism,
};

struct Action {
    ActionType type;
    int x;
    int y;

    Action()=default;
    Action(ActionType _type, int _x, int _y): type(_type), x(_x), y(_y){}
};

#endif //THELIFEENGINECPP_ACTIONS_H
