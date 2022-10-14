//
// Created by spaceeye on 15.09.22.
//

#ifndef LIFEENGINEEXTENDED_LEGACYANATOMYMUTATIONLOGIC_H
#define LIFEENGINEEXTENDED_LEGACYANATOMYMUTATIONLOGIC_H


#include "../../Stuff/Vector2.h"
#include "../../PRNGS/lehmer64.h"
#include "../../GridBlocks/BaseGridBlock.h"
#include "../../Stuff/BlockTypes.hpp"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "Rotation.h"
#include "Anatomy.h"
#include <random>
#include <boost/unordered_map.hpp>
#include <utility>

class LegacyAnatomyMutationLogic {

    static void set_single_adjacent(int x, int y, int x_offset, int y_offset,
                                    boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                    boost::unordered::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space,
                                    const BaseGridBlock &block);

    static void set_single_diagonal_adjacent(int x, int y, int x_offset, int y_offset,
                                             boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                             boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                             boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space);

    static inline void serialize_killing_space(const boost::unordered::unordered_map<int, boost::unordered_map<int, bool>> &killing_space,
                                               std::vector<SerializedAdjacentSpaceContainer> &_killing_space);

    static inline void serialize_eating_space(const boost::unordered::unordered_map<int, boost::unordered_map<int, bool>> &eating_space,
                                              std::vector<SerializedAdjacentSpaceContainer> &_eating_space);

    static inline void serialize_organism_blocks(const boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                                 std::vector<SerializedOrganismBlockContainer> &_organism_blocks);

    static inline void serialize_producing_space(const boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, ProducerAdjacent>> &producing_space,
                                                 const std::vector<int> &num_producing_space,
                                                 std::vector<std::vector<SerializedAdjacentSpaceContainer>> &_producing_space);

    template<typename T>
    static int get_map_size(boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, T>> map);

public:
    static void create_eating_space(boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                    boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
                                    boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                    int32_t mouth_blocks);

    static void create_killing_space(boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                     boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& killing_space,
                                     boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                     int32_t killer_blocks);

    static void create_producing_space(boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
                                       boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, ProducerAdjacent>>& producing_space,
                                       boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
                                       std::vector<int> & num_producing_space,
                                       int32_t producer_blocks);

    static void create_single_adjacent_space(
            boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
            boost::unordered::unordered_map<int, boost::unordered_map<int, bool>> &single_adjacent_space);

    static void create_single_diagonal_adjacent_space(
            boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
            boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_adjacent_space,
            boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& single_diagonal_adjacent_space);

    static SerializedOrganismStructureContainer * serialize(
            const boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, BaseGridBlock>> &organism_blocks,
            const boost::unordered::unordered_map<int, boost::unordered::unordered_map<int, ProducerAdjacent>>& producing_space,
            const boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& eating_space,
            const boost::unordered::unordered_map<int, boost::unordered_map<int, bool>>& killing_space,

            const std::vector<int> & num_producing_space,

            int32_t mouth_blocks,
            int32_t producer_blocks,
            int32_t mover_blocks,
            int32_t killer_blocks,
            int32_t armor_blocks,
            int32_t eye_blocks);

    static inline void serialize_eye_blocks(const std::vector<SerializedOrganismBlockContainer> &organism_blocks,
                                            std::vector<SerializedOrganismBlockContainer> &eye_blocks_vector,
                                            int eye_blocks);
};


#endif //LIFEENGINEEXTENDED_LEGACYANATOMYMUTATIONLOGIC_H
