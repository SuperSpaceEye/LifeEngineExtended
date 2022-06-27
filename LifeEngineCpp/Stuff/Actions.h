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
};

#endif //THELIFEENGINECPP_ACTIONS_H
