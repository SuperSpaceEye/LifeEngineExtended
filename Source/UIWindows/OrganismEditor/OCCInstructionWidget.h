//
// Created by spaceeye on 23.09.22.
//

#ifndef LIFEENGINEEXTENDED_OCCINSTRUCTIONWIDGET_H
#define LIFEENGINEEXTENDED_OCCINSTRUCTIONWIDGET_H

#include "OCCInstructionWidgetUI.h"
#include "EditorUI.h"
#include "../../Organism/CPU/OrganismConstructionCodeInstruction.h"

class OCCInstructionWidget: public QWidget {
    Q_OBJECT
private:
    Ui::OCCInstWidget ui;
    Ui::Editor & parent_ui;
public:
    OCCInstruction instruction;
    OCCInstructionWidget(bool first_widget, Ui::Editor & parent_ui, OCCInstruction start_instruction, QWidget * parent): parent_ui(parent_ui) {
        if (first_widget) {
            init_gui();
            ui.b_insert_left->hide();
            ui.b_delete_inst->hide();
            ui.cmb_occ_instructions->hide();

            auto pos = ui.frame->layout()->indexOf(ui.cmb_occ_instructions);

            auto label = new QLabel(QString::fromStdString(OCC_INSTRUCTIONS_NAME[(int)start_instruction]), ui.frame);
            this->ui.verticalLayout_3->insertWidget(pos-1, label);
            label->show();
        }
        setParent(parent);
    }

    void init_gui() {
        ui.setupUi(this);

        for (auto & name: OCC_INSTRUCTIONS_NAME) {
            ui.cmb_occ_instructions->addItem(QString::fromStdString(name));
        }
    }
private slots:
    void b_insert_right_slot() {
        auto pos = parent_ui.occ_layout->indexOf(this);
        parent_ui.occ_layout->insertWidget(pos, new OCCInstructionWidget(false, parent_ui, OCCInstruction::SetBlockMouth, parentWidget()));
    };
    void b_insert_left_slot() {
        auto pos = parent_ui.occ_layout->indexOf(this);
        parent_ui.occ_layout->insertWidget(pos-1, new OCCInstructionWidget(false, parent_ui, OCCInstruction::SetBlockMouth, parentWidget()));
    }
    void b_delete_inst_slot() {
        this->deleteLater();
    }

    void cmb_occ_instructions_slot(QString name) {
        auto sname = name.toStdString();
        instruction = static_cast<OCCInstruction>(get_index_of_occ_instruction_name(sname));
    }
};

#endif //LIFEENGINEEXTENDED_OCCINSTRUCTIONWIDGET_H
