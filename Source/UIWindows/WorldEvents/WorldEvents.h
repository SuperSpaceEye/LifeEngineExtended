//
// Created by spaceeye on 01.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTS_H
#define LIFEENGINEEXTENDED_WORLDEVENTS_H

#include <QDialogButtonBox>
#include <QSpacerItem>

#include "UIWindows/WorldEvents/EventNodesWidgets/ConditionalEventNodeWidget.h"
#include "UIWindows/WorldEvents/EventNodesWidgets/ChangeValueEventNodeWidget.h"
#include "UIWindows/MainWindow/WindowUI.h"
#include "UIWindows/WorldEvents/Misc/WorldEventsEnums.h"
#include "UIWindows/WorldEvents/Misc/ParametersList.h"

#include "SimulationEngine/SimulationEngine.h"
#include "Stuff/UIMisc.h"

#include "WorldEventsUI.h"
#include "EventNodes.h"

class WorldEvents: public QWidget {
    Q_OBJECT
public:
    enum class VerifyNodesFailCodes {
        NoProblems,
        UnknownNode,
        EmptyStartingNode,
        ChangeValueNodeNoChangeValue,
        ChangeValueIncorrectValue,
        ConditionalNodeNoValueToCheck,
        ConditionalNodeIncorrectValueToChange
    };
private:
    Ui::WorldEvents ui{};
    Ui::MainWindow * parent_ui = nullptr;
    ParametersList pl;
    EngineControlParameters * ecp = nullptr;
    SimulationEngine * engine = nullptr;

    std::vector<WorldEventNodes::BaseEventNode*> starting_nodes_container;
    std::vector<char*> repeating_branch;

    void closeEvent(QCloseEvent * event) override;

    void button_add_new_branch();
    QWidget * node_chooser(QHBoxLayout * widget_layout);

    VerifyNodesFailCodes verify_nodes();
    static WorldEvents::VerifyNodesFailCodes check_conditional_node(WorldEventNodes::BaseEventNode * node);
    static VerifyNodesFailCodes check_change_value_node(WorldEventNodes::BaseEventNode * node);
    WorldEventNodes::BaseEventNode *copy_node(WorldEventNodes::BaseEventNode *node, std::vector<WorldEventNodes::BaseEventNode *> &node_storage);
    void copy_nodes(std::vector<WorldEventNodes::BaseEventNode *> &start_nodes, std::vector<WorldEventNodes::BaseEventNode *> &node_storage);
public:
    WorldEvents(Ui::MainWindow * parent_ui,
                SimulationParameters * sp,
                OrganismBlockParameters * bp,
                OrganismInfoContainer * ic,
                EngineControlParameters * ecp,
                SimulationEngine * engine);

private slots:
    void b_apply_events_slot();
    void b_pause_events_slot();
    void b_resume_events_slot();
    void b_start_events_slot();
    void b_stop_events_slot();
    void b_stop_events_no_setting_reset_slot();

    void le_collect_info_every_n_slot();
    void le_update_world_events_every_n_slot();
};


#endif //LIFEENGINEEXTENDED_WORLDEVENTS_H
