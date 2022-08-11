//
// Created by spaceeye on 01.08.22.
//

#ifndef LIFEENGINEEXTENDED_WORLDEVENTS_H
#define LIFEENGINEEXTENDED_WORLDEVENTS_H

#include <QDialogButtonBox>
#include <QSpacerItem>

#include "WorldEventsUI.h"
#include "../MainWindow/WindowUI.h"
#include "EventNodes.h"
#include "ParametersList.h"
#include "../SimulationEngine/SimulationEngine.h"

#include "ConditionalEventNodeWidget.h"
#include "ChangeValueEventNodeWidget.h"
#include "../Stuff/MiscFuncs.h"

class WorldEvents: public QWidget {
    Q_OBJECT
private:
    Ui::WorldEvents ui{};
    Ui::MainWindow * parent_ui = nullptr;
    ParametersList pl;
    EngineControlParameters * ecp = nullptr;
    SimulationEngine * engine = nullptr;

    std::vector<BaseEventNode*> event_node_branch_starting_node_container;

    void closeEvent(QCloseEvent * event) override;

    void button_add_new_branch();
    QWidget * node_chooser(QHBoxLayout * widget_layout);

    bool verify_nodes();
    bool check_conditional_node(BaseEventNode * node);
    static bool check_change_value_node(BaseEventNode * node);
    BaseEventNode *copy_node(BaseEventNode *node, std::vector<BaseEventNode *> &node_storage);
    void copy_nodes(std::vector<BaseEventNode *> &start_nodes, std::vector<BaseEventNode *> &node_storage);
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
};


#endif //LIFEENGINEEXTENDED_WORLDEVENTS_H
