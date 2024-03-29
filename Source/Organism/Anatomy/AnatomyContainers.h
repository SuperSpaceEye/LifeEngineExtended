// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

//
// Created by spaceeye on 31.01.23.
//

#ifndef LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H
#define LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H

#include <utility>
#include <vector>
#include <string_view>

#include "Stuff/external/ArrayView.h"

#include "Stuff/structs/Vector2.h"
#include "Stuff/enums/BlockTypes.hpp"
#include "Stuff/enums/Rotation.h"
#include "Containers/OrganismBlockParameters.h"

#include "AnatomyCountersMap.h"

struct BaseSerializedContainer {
public:
    int relative_x;
    int relative_y;

    BaseSerializedContainer()=default;
    BaseSerializedContainer(int relative_x, int relative_y):
    relative_x(relative_x), relative_y(relative_y) {}

    inline Vector2<int> get_pos(Rotation rotation) const {
        return std::array<Vector2<int>, 4> {
                Vector2<int>{relative_x, relative_y},
                Vector2<int>{relative_y, -relative_x},
                Vector2<int>{-relative_x, -relative_y},
                Vector2<int>{-relative_y, relative_x}
        }[int(rotation)];
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
    Rotation get_block_rotation_on_grid(Rotation organism_rotation) const {
        uint_fast8_t new_int_rotation = static_cast<uint_fast8_t>(organism_rotation) + static_cast<uint_fast8_t>(rotation);
        return static_cast<Rotation>(new_int_rotation%4);
    }

    inline float get_food_cost(const OrganismBlockParameters &bp) const {
        return bp.pa[(int)type-1].food_cost;
    }

    inline bool operator==(const SerializedOrganismBlockContainer & other) {
        return
           type==other.type
        && rotation==other.rotation
        && relative_x==other.relative_x
        && relative_y==other.relative_y;
    }
};

struct SerializedAdjacentSpaceContainer: BaseSerializedContainer {
    uint8_t num = 1;
    SerializedAdjacentSpaceContainer()=default;
    SerializedAdjacentSpaceContainer(int relative_x, int relative_y):
            BaseSerializedContainer(relative_x, relative_y) {}
    SerializedAdjacentSpaceContainer(int relative_x, int relative_y, uint8_t count):
            BaseSerializedContainer(relative_x, relative_y), num(count) {}
};

struct SerializedOrganismStructureContainer {
    std::vector<SerializedOrganismBlockContainer> organism_blocks;
    std::vector<std::vector<SerializedAdjacentSpaceContainer>> producing_space;
    std::vector<SerializedAdjacentSpaceContainer> eating_space;
    std::vector<SerializedAdjacentSpaceContainer> killing_space;
    std::vector<SerializedOrganismBlockContainer> eye_block_vec;

    std::vector<uint32_t> eating_mask;
    std::vector<uint32_t> killer_mask;

    AnatomyCounters<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES> c = make_anatomy_counters();

    SerializedOrganismStructureContainer()=default;
    SerializedOrganismStructureContainer(
            std::vector<SerializedOrganismBlockContainer> organism_blocks,
            std::vector<std::vector<SerializedAdjacentSpaceContainer>> producing_space,
            std::vector<SerializedAdjacentSpaceContainer> eating_space,
            std::vector<SerializedAdjacentSpaceContainer> killing_space,
            std::vector<SerializedOrganismBlockContainer> eye_block_vector,
            std::vector<uint32_t> eating_mask,
            std::vector<uint32_t> killer_mask,

            AnatomyCounters<int, NUM_ORGANISM_BLOCKS, (std::string_view*)SW_ORGANISM_BLOCK_NAMES> c
            ):
            organism_blocks                (std::move(organism_blocks)),
            producing_space                (std::move(producing_space)),
            eating_space                   (std::move(eating_space)),
            killing_space                  (std::move(killing_space)),
            eye_block_vec                  (std::move(eye_block_vector)),

            eating_mask(std::move(eating_mask)),
            killer_mask(std::move(killer_mask)),

            c(std::move(c))
            {}
    void move_s(SerializedOrganismStructureContainer *structure) {
        organism_blocks = std::move(structure->organism_blocks);

        producing_space = std::move(structure->producing_space);
        eating_space    = std::move(structure->eating_space);
        killing_space   = std::move(structure->killing_space);
        eye_block_vec   = std::move(structure->eye_block_vec);

        eating_mask = std::move(structure->eating_mask);
        killer_mask = std::move(structure->killer_mask);

        c = structure->c;
    }
    void copy_s(const SerializedOrganismStructureContainer * structure) {
        organism_blocks = std::vector(structure->organism_blocks);

        producing_space = std::vector(structure->producing_space);
        eating_space    = std::vector(structure->eating_space);
        killing_space   = std::vector(structure->killing_space);
        eye_block_vec   = std::vector(structure->eye_block_vec);

        eating_mask = std::vector(structure->eating_mask);
        killer_mask = std::vector(structure->killer_mask);

        c = structure->c;
    }
};

#endif //LIFEENGINEEXTENDED_ANATOMYCONTAINERS_H
