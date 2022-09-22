//
// Created by spaceeye on 17.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCPARAMETERS_H
#define LIFEENGINEEXTENDED_OCCPARAMETERS_H

#include "OCCParametersUI.h"
#include "../MainWindow/WindowUI.h"
#include "../../Containers/CPU/OrganismConstructionCodeParameters.h"
#include "../../Stuff/MiscFuncs.h"
#include "../../Organism/CPU/OrganismConstructionCodeInstruction.h"
#include "../../SimulationEngine/SimulationEngine.h"

class OCCParametersWindow: public QWidget {
    Q_OBJECT
private:
    Ui::OCCParametes ui{};
    Ui::MainWindow * parent_ui = nullptr;

    OCCParameters & occp;
    SimulationEngine & engine;

    void closeEvent(QCloseEvent * event) override;

    void init_gui();

    void create_mutation_type_distribution();
    void create_group_size_distribution();
    void create_occ_instructions_distribution();
    void create_move_distance_distribution();

    static QLayout *prepare_layout(QLayout *layout) ;
public:
    OCCParametersWindow(Ui::MainWindow * parent_ui, OCCParameters & occp, SimulationEngine & engine);

    void reinit_gui();
private slots:
    void cb_use_uniform_group_size_slot(bool state);
    void cb_use_uniform_move_distance_slot(bool state);
    void cb_use_uniform_mutation_type_slot(bool state);
    void cb_use_uniform_occ_instructions_slot(bool state);

    void le_max_group_size_slot();
    void le_max_move_distance_slot();
};


#endif //LIFEENGINEEXTENDED_OCCPARAMETERS_H
