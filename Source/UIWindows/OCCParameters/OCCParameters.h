//
// Created by spaceeye on 17.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCPARAMETERS_H
#define LIFEENGINEEXTENDED_OCCPARAMETERS_H

#include <QKeyEvent>

#include "UIWindows/MainWindow/WindowUI.h"

#include "Organism/OCC/OrganismConstructionCodeInstruction.h"
#include "Organism/OCC/OrganismConstructionCodeInstruction.h"
#include "Containers/OrganismConstructionCodeParameters.h"
#include "Stuff/UIMisc.h"
#include "SimulationEngine/SimulationEngine.h"

#include "OCCParametersUI.h"

class OCCParametersWindow: public QWidget {
    Q_OBJECT
private:
    Ui::OCCParametes ui{};
    Ui::MainWindow * parent_ui = nullptr;

    OCCParameters & occp;
    SimulationEngine & engine;

    void closeEvent(QCloseEvent * event) override;
    void keyPressEvent(QKeyEvent * event) override {
        if (event->key() == Qt::Key_Escape) {
            close();
        }
    }

    void init_gui();

    void create_mutation_type_distribution();
    void create_group_size_distribution();
    void create_occ_instructions_distribution(bool no_reset = false);
    void create_move_distance_distribution();

    static QLayout *prepare_layout(QLayout *layout);

public:
    OCCParametersWindow(Ui::MainWindow * parent_ui, OCCParameters & occp, SimulationEngine & engine);

    void reinit_gui(bool just_reinit_gui = false);
private slots:
    void cb_use_uniform_group_size_slot(bool state);
    void cb_use_uniform_move_distance_slot(bool state);
    void cb_use_uniform_mutation_type_slot(bool state);
    void cb_use_uniform_occ_instructions_slot(bool state);

    void le_max_group_size_slot();
    void le_max_move_distance_slot();
};


#endif //LIFEENGINEEXTENDED_OCCPARAMETERS_H
