//
// Created by spaceeye on 17.09.22.
//

#include "OCCParameters.h"

OCCParametersWindow::OCCParametersWindow(Ui::MainWindow *parent_ui, OCCParameters & occp, SimulationEngine &engine):
    parent_ui(parent_ui), occp(occp), engine(engine) {
    ui.setupUi(this);
    init_gui();
    create_occ_instructions_distribution();
    create_mutation_type_distribution();
    create_move_distance_distribution();
    create_group_size_distribution();
}

void OCCParametersWindow::closeEvent(QCloseEvent *event) {
    parent_ui->tb_open_occ_parameters->setChecked(false);
    QWidget::closeEvent(event);
}

void OCCParametersWindow::init_gui() {
    ui.cb_use_uniform_mutation_type   ->setChecked(occp.uniform_mutation_distribution);
    ui.cb_use_uniform_group_size      ->setChecked(occp.uniform_group_size_distribution);
    ui.cb_use_uniform_occ_instructions->setChecked(occp.uniform_occ_instructions_mutation);
    ui.cb_use_uniform_move_distance   ->setChecked(occp.uniform_move_distance);

    ui.le_max_group_size   ->setText(QString::fromStdString(std::to_string(occp.max_group_size)));
    ui.le_max_move_distance->setText(QString::fromStdString(std::to_string(occp.max_distance)));
}

QLayout *OCCParametersWindow::prepare_layout(QLayout *layout) {
    QLayoutItem * item;
    while ((item = layout->takeAt(0)) != nullptr) {
        QLayoutItem * item2;
        while ((item2 = item->layout()->takeAt(0)) != nullptr) {
            item2->widget()->deleteLater();
            delete item2;
        }
        layout->removeItem(item);
        item->layout()->deleteLater();
        delete item;
    }

    return layout;
}

void OCCParametersWindow::create_mutation_type_distribution() {
    auto *layout = ui.mutation_types_widget->layout();

    std::array<std::string, 5> mutation_types{
        "Append Random",
        "Insert Random",
        "Change Random",
        "Delete Random",
        "Move Random"
    };

    for (int i = 0; i < occp.mutation_type_weights.size(); i++) {
        auto * new_layout = new QHBoxLayout{};
        auto * label = new QLabel(QString::fromStdString("\""+mutation_types[i]+"\" weight "), ui.mutation_types_widget);
        auto * line_edit = new QLineEdit(QString::fromStdString("1"), ui.mutation_types_widget);

        auto * p_occp = &occp;
        auto * p_engine = &engine;

        connect(line_edit, &QLineEdit::returnPressed, [i, p_occp, line_edit, p_engine](){
            auto & occp = *p_occp;
            auto & engine = *p_engine;
            le_slot_lower_bound<int>(occp.mutation_type_weights[i], occp.mutation_type_weights[i], "int", line_edit, 0, "0");
            engine.pause();
            occp.mutation_discrete_distribution = std::discrete_distribution<int>(occp.mutation_type_weights.begin(), occp.mutation_type_weights.end());
            engine.unpause();
        });

        new_layout->addWidget(label);
        new_layout->addWidget(line_edit);

        label->show();
        line_edit->show();

        layout->addItem(new_layout);
    }
}

void OCCParametersWindow::create_group_size_distribution() {
    auto *layout = prepare_layout(ui.group_size_layout);

    for (int i = 0; i < occp.group_size_weights.size(); i++) {
        auto * new_layout = new QHBoxLayout{};
        auto * label = new QLabel(QString::fromStdString("\""+std::to_string(i+1)+"\" weight "), ui.group_size_widget);
        auto * line_edit = new QLineEdit(QString::fromStdString("1"), ui.group_size_widget);

        auto * p_occp = &occp;
        auto * p_engine = &engine;

        connect(line_edit, &QLineEdit::returnPressed, [i, p_occp, line_edit, p_engine](){
            auto & occp = *p_occp;
            auto & engine = *p_engine;
            le_slot_lower_bound<int>(occp.group_size_weights[i], occp.group_size_weights[i], "int", line_edit, 0, "0");
            engine.pause();
            occp.group_size_discrete_distribution = std::discrete_distribution<int>(occp.group_size_weights.begin(), occp.group_size_weights.end());
            engine.unpause();
        });

        new_layout->addWidget(label);
        new_layout->addWidget(line_edit);

        label->show();
        line_edit->show();

        layout->addItem(new_layout);
    }
}

void OCCParametersWindow::create_occ_instructions_distribution() {
    auto *layout = ui.occ_instructions_weights_layout;

    occp.occ_instructions_mutation_weights.resize(OCC_INSTRUCTIONS_NAME.size(), 1);

    for (int i = 0; i < OCC_INSTRUCTIONS_NAME.size(); i++) {
        auto * new_layout = new QHBoxLayout{};
        auto * label = new QLabel(QString::fromStdString("\""+OCC_INSTRUCTIONS_NAME[i])+"\" weight ", ui.occ_mutation_type_widget);
        auto * line_edit = new QLineEdit(QString::fromStdString("1"), ui.occ_mutation_type_widget);

        auto * p_occp = &occp;
        auto * p_engine = &engine;

        connect(line_edit, &QLineEdit::returnPressed, [i, p_occp, line_edit, p_engine](){
            auto & occp = *p_occp;
            auto & engine = *p_engine;
            le_slot_lower_bound<int>(occp.occ_instructions_mutation_weights[i], occp.occ_instructions_mutation_weights[i], "int", line_edit, 0, "0");
            engine.pause();
            occp.occ_instructions_mutation_discrete_distribution = std::discrete_distribution<int>(occp.occ_instructions_mutation_weights.begin(), occp.occ_instructions_mutation_weights.end());
            engine.unpause();
        });

        new_layout->addWidget(label);
        new_layout->addWidget(line_edit);

        label->show();
        line_edit->show();

        layout->addItem(new_layout);
    }
}

void OCCParametersWindow::create_move_distance_distribution() {
    auto *layout = prepare_layout(ui.move_distance_layout);

    for (int i = 0; i < occp.move_distance_mutation_weights.size(); i++) {
        auto * new_layout = new QHBoxLayout{};
        auto * label = new QLabel(QString::fromStdString("\""+std::to_string(i+1)+"\" weight "), ui.move_distance_widget);
        auto * line_edit = new QLineEdit(QString::fromStdString("1"), ui.move_distance_widget);

        auto * p_occp = &occp;
        auto * p_engine = &engine;

        connect(line_edit, &QLineEdit::returnPressed, [i, p_occp, line_edit, p_engine](){
            auto & occp = *p_occp;
            auto & engine = *p_engine;
            le_slot_lower_bound<int>(occp.move_distance_mutation_weights[i], occp.move_distance_mutation_weights[i], "int", line_edit, 0, "0");
            engine.pause();
            occp.move_distance_mutation_discrete_distribution = std::discrete_distribution<int>(occp.move_distance_mutation_weights.begin(), occp.move_distance_mutation_weights.end());
            engine.unpause();
        });

        new_layout->addWidget(label);
        new_layout->addWidget(line_edit);

        label->show();
        line_edit->show();

        layout->addItem(new_layout);
    }
}

void OCCParametersWindow::reinit_gui() {
    init_gui();
    create_occ_instructions_distribution();
    create_mutation_type_distribution();
    create_move_distance_distribution();
    create_group_size_distribution();
}