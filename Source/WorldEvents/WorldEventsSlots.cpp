//
// Created by spaceeye on 10.08.22.
//

#include "WorldEvents.h"

void WorldEvents::b_apply_events_slot() {
    std::vector<BaseEventNode*> start_nodes;
    std::vector<char> _repeating_branch;
    std::vector<BaseEventNode*> node_storage;

    _repeating_branch.reserve(repeating_branch.size());

    for (int i = 0; i < repeating_branch.size(); i++) {
        _repeating_branch[i] = *repeating_branch[i];
    }

    if (!verify_nodes()) {
        display_message("Applying new events failed");
        return;
    }

    copy_nodes(start_nodes, node_storage);

    engine->reset_world_events(std::move(start_nodes), std::move(_repeating_branch), std::move(node_storage));
    engine->start_world_events();
    b_resume_events_slot();
}

void WorldEvents::b_pause_events_slot() {
    ecp->pause_world_events = true;
    ui.world_events_status_label->setText(QString("Events paused."));
}

void WorldEvents::b_resume_events_slot() {
    ecp->pause_world_events = false;
    ui.world_events_status_label->setText(QString("Events running."));
}

void WorldEvents::b_start_events_slot() {
    if (ecp->execute_world_events) { return;}
    engine->start_world_events();
    ui.world_events_status_label->setText(QString("Events running."));
}

void WorldEvents::b_stop_events_slot() {
    if (!ecp->execute_world_events) { return;}
    engine->stop_world_events();
    ui.world_events_status_label->setText(QString("Events stopped."));
}