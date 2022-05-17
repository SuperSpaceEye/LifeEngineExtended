//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ANATOMY_H
#define THELIFEENGINECPP_ANATOMY_H

#include <utility>
#include <boost/unordered_map.hpp>
#include <random>

#include "Organism_parts/OrganismBlock.h"
#include "Rotation.h"
#include "../OrganismBlockParameters.h"
#include "../BlockTypes.hpp"

//TODO refactor this
//TODO this code is a mess and i became confused, so i don't really know if it will even work.

struct pos {
    int x;
    int y;
};

struct BaseSerializedContainer {
public:
    int relative_x;
    int relative_y;

    BaseSerializedContainer()=default;
    BaseSerializedContainer(int relative_x, int relative_y):
    relative_x(relative_x), relative_y(relative_y) {}

    pos get_pos(Rotation rotation) {
        switch (rotation) {
            case Rotation::UP:    return pos{ relative_x,  relative_y};
            case Rotation::LEFT:  return pos{-relative_y,  relative_x};
            case Rotation::DOWN:  return pos{ -relative_x, -relative_y};
            case Rotation::RIGHT: return pos{relative_y, -relative_x};
            default: return pos{relative_x, relative_y};
        }
    }
};

struct SerializedOrganismBlockContainer: BaseSerializedContainer {
    OrganismBlock organism_block;

    SerializedOrganismBlockContainer()=default;
    SerializedOrganismBlockContainer(OrganismBlock organism_block, int relative_x, int relative_y):
    organism_block(organism_block), BaseSerializedContainer(relative_x, relative_y) {}
};

struct SerializedAdjacentSpaceContainer: BaseSerializedContainer {
    SerializedAdjacentSpaceContainer()=default;
    SerializedAdjacentSpaceContainer(int relative_x, int relative_y):
    BaseSerializedContainer(relative_x, relative_y) {}
};

struct SerializedArmorSpaceContainer: BaseSerializedContainer {
    bool is_armored;
    SerializedArmorSpaceContainer()=default;
    SerializedArmorSpaceContainer(int relative_x, int relative_y, bool is_armored):
    BaseSerializedContainer(relative_x, relative_y), is_armored(is_armored) {}
};

struct SerializedOrganismStructureContainer {
    std::vector<SerializedOrganismBlockContainer> organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> producing_space;
    std::vector<SerializedAdjacentSpaceContainer> eating_space;
    std::vector<SerializedArmorSpaceContainer>    armor_space;

    std::vector<SerializedAdjacentSpaceContainer> single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> double_adjacent_space;

    int32_t mouth_blocks{};
    int32_t producer_blocks{};
    int32_t mover_blocks{};
    int32_t killer_blocks{};
    int32_t armor_blocks{};
    int32_t eye_blocks{};

    SerializedOrganismStructureContainer()=default;
    SerializedOrganismStructureContainer(
            std::vector<SerializedOrganismBlockContainer> organism_blocks,
            std::vector<SerializedAdjacentSpaceContainer> producing_space,
            std::vector<SerializedAdjacentSpaceContainer> eating_space,
            std::vector<SerializedArmorSpaceContainer>    armor_space,
            std::vector<SerializedAdjacentSpaceContainer> single_adjacent_space,
            std::vector<SerializedAdjacentSpaceContainer> single_diagonal_adjacent_space,
            std::vector<SerializedAdjacentSpaceContainer> double_adjacent_space,
            int32_t mouth_blocks,
            int32_t producer_blocks,
            int32_t mover_blocks,
            int32_t killer_blocks,
            int32_t armor_blocks,
            int32_t eye_blocks):
            organism_blocks                (std::move(organism_blocks)),
            producing_space                (std::move(producing_space)),
            eating_space                   (std::move(eating_space)),
            armor_space                    (std::move(armor_space)),

            single_adjacent_space          (std::move(single_adjacent_space)),
            single_diagonal_adjacent_space (std::move(single_diagonal_adjacent_space)),
            double_adjacent_space          (std::move(double_adjacent_space)),
            mouth_blocks(mouth_blocks),
            producer_blocks(producer_blocks),
            mover_blocks(mover_blocks),
            killer_blocks(killer_blocks),
            armor_blocks(armor_blocks),
            eye_blocks(eye_blocks)
            {}
};

class Anatomy {
private:
    static void set_single_adjacent(int x, int y, int x_offset, int y_offset,
                             boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space);

    static void set_double_adjacent(int x, int y, int x_offset, int y_offset,
                             boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                             boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space);

    static void create_single_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space);

    static void create_single_diagonal_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                               boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                               boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space);

    static void create_double_adjacent_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                      boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space);

    static void create_producing_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                boost::unordered_map<int, boost::unordered_map<int, bool>>& producing_space,
                                boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                int32_t producer_blocks);

    static void create_eating_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                       boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
                                       boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                       int32_t mouth_blocks);

    static void create_armor_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                   boost::unordered_map<int, boost::unordered_map<int, bool>>& armor_space,
                                   boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                   int32_t armor_blocks);

    static void reset_organism_center(std::vector<SerializedOrganismBlockContainer> & _organism_blocks,
                                      boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> & organism_blocks,
                                      int & x, int & y);

    static SerializedOrganismStructureContainer * serialize(const boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& producing_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& armor_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space,
                                                            int32_t mouth_blocks,
                                                            int32_t producer_blocks,
                                                            int32_t mover_blocks,
                                                            int32_t killer_blocks,
                                                            int32_t armor_blocks,
                                                            int32_t eye_blocks);

//    DeserializedOrganismStructureContainer deserialize(std::vector<SerializedOrganismBlockContainer>& organism_blocks,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& single_adjacent_space,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& single_diagonal_adjacent_space,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& double_adjacent_space);

    template<typename T>
    static int get_map_size(boost::unordered_map<int, boost::unordered_map<int, T>> map);

    SerializedOrganismStructureContainer * add_block(BlockTypes type, int block_choice = -1, int x_ = 0, int y_ = 0);
    SerializedOrganismStructureContainer * change_block(BlockTypes type, int block_choice);
    SerializedOrganismStructureContainer * remove_block(int block_choice);

public:
    std::vector<SerializedOrganismBlockContainer> _organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> _producing_space;
    std::vector<SerializedAdjacentSpaceContainer> _eating_space;
    std::vector<SerializedArmorSpaceContainer>    _armor_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _double_adjacent_space;

    int32_t _mouth_blocks    = 0;
    int32_t _producer_blocks = 0;
    int32_t _mover_blocks    = 0;
    int32_t _killer_blocks   = 0;
    int32_t _armor_blocks    = 0;
    int32_t _eye_blocks      = 0;

    explicit Anatomy(SerializedOrganismStructureContainer *structure);
    explicit Anatomy(const std::shared_ptr<Anatomy>& anatomy);
    Anatomy()=default;

    ~Anatomy();

    SerializedOrganismStructureContainer * add_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt);
    SerializedOrganismStructureContainer * change_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt);
    SerializedOrganismStructureContainer * remove_random_block(std::mt19937& mt);

    void set_block(BlockTypes type, int x, int y);
};


#endif //THELIFEENGINECPP_ANATOMY_H
