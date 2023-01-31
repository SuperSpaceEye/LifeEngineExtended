// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 31.01.23.
//

#ifndef LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H
#define LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H

#include <vector>
#include <string_view>

#include "../../Stuff/Vector2.h"
#include "../../Stuff/BlockTypes.hpp"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "Rotation.h"
#include "../../Stuff/ConstMap.h"

struct ProducerAdjacent {
    int producer = -1;
};

struct BaseSerializedContainer {
public:
    int relative_x;
    int relative_y;

    BaseSerializedContainer()=default;
    BaseSerializedContainer(int relative_x, int relative_y):
    relative_x(relative_x), relative_y(relative_y) {}

    inline Vector2<int> get_pos(Rotation rotation) {
        switch (rotation) {
            case Rotation::UP:    return Vector2<int>{relative_x, relative_y};
            case Rotation::LEFT:  return Vector2<int>{relative_y, -relative_x};
            case Rotation::DOWN:  return Vector2<int>{-relative_x, -relative_y};
            case Rotation::RIGHT: return Vector2<int>{-relative_y, relative_x};
            default: return Vector2<int>{relative_x, relative_y};
        }
    }
};

struct SerializedOrganismBlockContainer: BaseSerializedContainer {
    BlockTypes type;
    //for now only for eye BLOCK_NAMES
    //local rotation of a block
    Rotation rotation = Rotation::UP;

    SerializedOrganismBlockContainer()=default;
    SerializedOrganismBlockContainer(BlockTypes type, Rotation rotation, int relative_x, int relative_y):
            BaseSerializedContainer(relative_x, relative_y), type(type), rotation(rotation) {}
    Rotation get_block_rotation_on_grid(Rotation organism_rotation) {
        uint_fast8_t new_int_rotation = static_cast<uint_fast8_t>(organism_rotation) + static_cast<uint_fast8_t>(rotation);
        return static_cast<Rotation>(new_int_rotation%4);
    }

    float get_food_cost(OrganismBlockParameters &bp) const {
        return bp.pa[(int)type-1].food_cost;
    }
};

struct SerializedAdjacentSpaceContainer: BaseSerializedContainer {
    SerializedAdjacentSpaceContainer()=default;
    SerializedAdjacentSpaceContainer(int relative_x, int relative_y):
    BaseSerializedContainer(relative_x, relative_y) {}
};

struct SerializedOrganismStructureContainer {
    std::vector<SerializedOrganismBlockContainer> organism_blocks;
    std::vector<std::vector<SerializedAdjacentSpaceContainer>> producing_space;
    std::vector<SerializedAdjacentSpaceContainer> eating_space;
    std::vector<SerializedAdjacentSpaceContainer> killing_space;
    std::vector<SerializedOrganismBlockContainer> eye_blocks_vec;

    ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES> c = get_map();

    SerializedOrganismStructureContainer()=default;
    SerializedOrganismStructureContainer(
            std::vector<SerializedOrganismBlockContainer> organism_blocks,
            std::vector<std::vector<SerializedAdjacentSpaceContainer>> producing_space,
            std::vector<SerializedAdjacentSpaceContainer> eating_space,
            std::vector<SerializedAdjacentSpaceContainer> killing_space,
            std::vector<SerializedOrganismBlockContainer> eye_block_vector,

            ConstMap<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES> c
            ):
            organism_blocks                (std::move(organism_blocks)),
            producing_space                (std::move(producing_space)),
            eating_space                   (std::move(eating_space)),
            killing_space                  (std::move(killing_space)),
            eye_blocks_vec                 (std::move(eye_block_vector)),

            c(c)
            {}
};

#endif //LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H
