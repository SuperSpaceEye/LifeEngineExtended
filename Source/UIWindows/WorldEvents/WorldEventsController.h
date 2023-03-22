//
// Created by spaceeye on 07.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTSCONTROLLER_H
#define LIFEENGINEEXTENDED_WORLDEVENTSCONTROLLER_H

#include <vector>
#include <cstdint>

#include "EventNodes.h"

class WorldEventsController {
private:
    //holds starting nodes for each event branch
    std::vector<WorldEventNodes::BaseEventNode*> start_nodes{};
    //holds cursors for each event branch.
    std::vector<WorldEventNodes::BaseEventNode*> node_cursors{};
    //corresponds to cursors. If true, when cursor hits nullptr will return to start node. If false, will stop execution.
    std::vector<char> repeating_branch{};
    //holds pointers to each node.
    std::vector<WorldEventNodes::BaseEventNode*> node_storage{};

    void delete_all_nodes();
public:
    WorldEventsController()=default;

    void tick_events(uint64_t time_point, bool pause_events);

    void reset_events(std::vector<WorldEventNodes::BaseEventNode *> _start_nodes,
                      std::vector<char> _repeating_branch,
                      std::vector<WorldEventNodes::BaseEventNode *> _node_storage);

    void reset();
};


#endif //LIFEENGINEEXTENDED_WORLDEVENTSCONTROLLER_H
