//
// Created by spaceeye on 07.08.22.
//

#include "WorldEventsController.h"
#include "WorldEventsEnums.h"

void WorldEventsController::delete_all_nodes() {
    for (auto & node: node_storage) {
        delete node;
    }

    start_nodes.clear();
    node_cursors.clear();
    repeating_branch.clear();
    node_storage.clear();
}

void WorldEventsController::reset_events(std::vector<BaseEventNode *> _start_nodes,
                                         std::vector<char> _repeating_branch,
                                         std::vector<BaseEventNode *> _node_storage) {
    delete_all_nodes();

    start_nodes = std::move(_start_nodes);
    node_cursors = std::vector(start_nodes);
    repeating_branch = std::move(_repeating_branch);
    node_storage = std::move(_node_storage);
}

void WorldEventsController::tick_events(uint64_t time_point, bool pause_events) {
    //choosing events in reverse because events should be applied in reverse order of importance.
    for (int i = node_cursors.size()-1; i >= 0; i--) {
        auto & node = node_cursors[i];
        if (node == nullptr) {
            if (!repeating_branch[i]) {
                continue;
            }
            node = start_nodes[i];
        }
        node = node->update(time_point, pause_events);
    }
}

void WorldEventsController::reset() {
    for (auto & node: node_cursors) {
        //If execution of event stopped abruptly.
        if (node != nullptr && node->type == NodeType::ChangeValue) {
            dynamic_cast<ChangeValueEventNode<float>*>(node)->reset_node();
        }
    }

    node_cursors = std::vector(start_nodes);
}