//
// Created by spaceeye on 12.03.23.
//

#ifndef LIFEENGINEEXTENDED_ACTIONTYPE_H
#define LIFEENGINEEXTENDED_ACTIONTYPE_H

enum class ActionType {
    TryAddFood,
    TryRemoveFood,
    TryAddWall,
    TryRemoveWall,
    TryAddOrganism,
    TryKillOrganism,
    TrySelectOrganism,

    DebugDisplayInfo,
};
#endif //LIFEENGINEEXTENDED_ACTIONTYPE_H
