//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_ANATOMY_H
#define THELIFEENGINECPP_ANATOMY_H

#include <utility>

#include "boost/unordered_map.hpp"
#include "Organism_parts/OrganismBlock.h"
#include "random"
#include "../OrganismBlockParameters.h"
#include "../BlockTypes.h"

//TODO refactor this
//TODO this code is a mess and i became confused, so i don't really know if it will even work.

struct SerializedOrganismBlockContainer {
    OrganismBlock organism_block;
    int relative_x;
    int relative_y;

    SerializedOrganismBlockContainer()=default;
    SerializedOrganismBlockContainer(OrganismBlock organism_block, int relative_x, int relative_y):
    organism_block(organism_block), relative_x(relative_x), relative_y(relative_y) {}
};

struct SerializedAdjacentSpaceContainer {
    int relative_x;
    int relative_y;

    SerializedAdjacentSpaceContainer()=default;
    SerializedAdjacentSpaceContainer(int relative_x, int relative_y):
    relative_x(relative_x), relative_y(relative_y) {}
};

struct SerializedOrganismStructureContainer {
    std::vector<SerializedOrganismBlockContainer> organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> producing_space;
    std::vector<SerializedAdjacentSpaceContainer> eating_space;
    std::vector<SerializedAdjacentSpaceContainer> single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> double_adjacent_space;

    int32_t producers{};
    int32_t movers{};

    SerializedOrganismStructureContainer()=default;
    SerializedOrganismStructureContainer(
            std::vector<SerializedOrganismBlockContainer> organism_blocks,
            std::vector<SerializedAdjacentSpaceContainer> producing_space,
            std::vector<SerializedAdjacentSpaceContainer> eating_space,
            std::vector<SerializedAdjacentSpaceContainer> single_adjacent_space,
            std::vector<SerializedAdjacentSpaceContainer> single_diagonal_adjacent_space,
            std::vector<SerializedAdjacentSpaceContainer> double_adjacent_space,
            int32_t producers, int32_t movers):
    organism_blocks                (std::move(organism_blocks)),
    producing_space                (std::move(producing_space)),
    eating_space                   (std::move(eating_space)),
    single_adjacent_space          (std::move(single_adjacent_space)),
    single_diagonal_adjacent_space (std::move(single_diagonal_adjacent_space)),
    double_adjacent_space          (std::move(double_adjacent_space)),
    producers(producers), movers(movers) {}
};

//struct DeserializedOrganismStructureContainer {
////    std::vector<SerializedAdjacentSpaceContainer> producing_space
//    boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space;
//    boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space;
//
//    int32_t producers;
//    int32_t movers;
//
//    DeserializedOrganismStructureContainer()=default;
//    DeserializedOrganismStructureContainer(
//            boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>> organism_blocks,
//            boost::unordered_map<int, boost::unordered_map<int, bool>> producing_space,
//            boost::unordered_map<int, boost::unordered_map<int, bool>> single_adjacent_space,
//            boost::unordered_map<int, boost::unordered_map<int, bool>> single_diagonal_adjacent_space,
//            boost::unordered_map<int, boost::unordered_map<int, bool>> double_adjacent_space,
//            int32_t producers, int32_t movers):
//    organism_blocks                (std::move(organism_blocks)),
//    producing_space                (std::move(producing_space)),
//    single_adjacent_space          (std::move(single_adjacent_space)),
//    single_diagonal_adjacent_space (std::move(single_diagonal_adjacent_space)),
//    double_adjacent_space          (std::move(double_adjacent_space)),
//    producers(producers), movers(movers) {}
//};

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
                                int32_t producers);

    static void create_eating_space(boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                       boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
                                       boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space);

    static SerializedOrganismStructureContainer * serialize(const boost::unordered_map<int, boost::unordered_map<int, OrganismBlock>>& organism_blocks,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& producing_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space,
                                                            const boost::unordered_map<int, boost::unordered_map<int, bool>>& double_adjacent_space,
                                                            int32_t producers, int32_t movers);

//    DeserializedOrganismStructureContainer deserialize(std::vector<SerializedOrganismBlockContainer>& organism_blocks,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& single_adjacent_space,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& single_diagonal_adjacent_space,
//                                                       std::vector<SerializedAdjacentSpaceContainer>& double_adjacent_space);

    SerializedOrganismStructureContainer * add_block(BlockTypes type, int block_choice = -1, int x_ = 0, int y_ = 0);
    SerializedOrganismStructureContainer * change_block(BlockTypes type, int block_choice);
    SerializedOrganismStructureContainer * remove_block(int block_choice);

public:
    std::vector<SerializedOrganismBlockContainer> _organism_blocks;
    std::vector<SerializedAdjacentSpaceContainer> _producing_space;
    std::vector<SerializedAdjacentSpaceContainer> _eating_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _single_diagonal_adjacent_space;
    std::vector<SerializedAdjacentSpaceContainer> _double_adjacent_space;

    int32_t _producers = 0;
    int32_t _movers = 0;

    SerializedOrganismStructureContainer * structure = nullptr;

    Anatomy(SerializedOrganismStructureContainer *structure);
    Anatomy(const Anatomy *anatomy);
    Anatomy()=default;
    ~Anatomy();

    SerializedOrganismStructureContainer * add_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt);
    SerializedOrganismStructureContainer * change_random_block(OrganismBlockParameters& block_parameters, std::mt19937& mt);
    SerializedOrganismStructureContainer * remove_random_block(std::mt19937& mt);

    void set_block(BlockTypes type, int x, int y);
};


#endif //THELIFEENGINECPP_ANATOMY_H
