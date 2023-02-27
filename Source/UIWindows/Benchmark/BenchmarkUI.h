/********************************************************************************
** Form generated from reading UI file 'benchmark.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef BENCHMARKUI_H
#define BENCHMARKUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Benchmark
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *main_tab;
    QVBoxLayout *verticalLayout_2;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_4;
    QSpacerItem *verticalSpacer_2;
    QPushButton *b_run_all_benchmarks;
    QSpacerItem *verticalSpacer_4;
    QPushButton *b_benchmark_produce_food;
    QPushButton *b_benchmark_eat_food;
    QPushButton *b_benchmark_apply_damage;
    QPushButton *b_benchmark_tick_lifetime;
    QPushButton *b_benchmark_get_observations;
    QPushButton *b_benchmark_think_decision;
    QPushButton *b_benchmark_rotate_organism;
    QPushButton *b_benchmark_move_organism;
    QPushButton *b_benchmark_try_make_child;
    QSpacerItem *verticalSpacer_3;
    QPushButton *b_stop_benchmarks;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout_5;
    QTextEdit *benchmarks_output_text_edit;
    QWidget *settings_tab;
    QVBoxLayout *verticalLayout_6;
    QSpacerItem *verticalSpacer_5;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QLineEdit *le_grid_width;
    QLabel *label;
    QLineEdit *le_grid_height;
    QPushButton *b_apply_grid_size;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QLineEdit *le_num_benchmark_organisms;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_4;
    QLineEdit *le_num_iterations;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_5;
    QLineEdit *le_organisms_diameter;
    QSpacerItem *verticalSpacer_6;

    void setupUi(QWidget *Benchmark)
    {
        if (Benchmark->objectName().isEmpty())
            Benchmark->setObjectName("Benchmark");
        Benchmark->resize(1031, 618);
        verticalLayout = new QVBoxLayout(Benchmark);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Benchmark);
        tabWidget->setObjectName("tabWidget");
        main_tab = new QWidget();
        main_tab->setObjectName("main_tab");
        verticalLayout_2 = new QVBoxLayout(main_tab);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(main_tab);
        scrollArea->setObjectName("scrollArea");
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName("scrollAreaWidgetContents");
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1025, 585));
        verticalLayout_3 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName("verticalLayout_3");
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout->setContentsMargins(-1, -1, 0, 0);
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName("verticalLayout_4");
        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_2);

        b_run_all_benchmarks = new QPushButton(scrollAreaWidgetContents);
        b_run_all_benchmarks->setObjectName("b_run_all_benchmarks");

        verticalLayout_4->addWidget(b_run_all_benchmarks);

        verticalSpacer_4 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_4);

        b_benchmark_produce_food = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_produce_food->setObjectName("b_benchmark_produce_food");

        verticalLayout_4->addWidget(b_benchmark_produce_food);

        b_benchmark_eat_food = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_eat_food->setObjectName("b_benchmark_eat_food");

        verticalLayout_4->addWidget(b_benchmark_eat_food);

        b_benchmark_apply_damage = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_apply_damage->setObjectName("b_benchmark_apply_damage");

        verticalLayout_4->addWidget(b_benchmark_apply_damage);

        b_benchmark_tick_lifetime = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_tick_lifetime->setObjectName("b_benchmark_tick_lifetime");

        verticalLayout_4->addWidget(b_benchmark_tick_lifetime);

        b_benchmark_get_observations = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_get_observations->setObjectName("b_benchmark_get_observations");

        verticalLayout_4->addWidget(b_benchmark_get_observations);

        b_benchmark_think_decision = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_think_decision->setObjectName("b_benchmark_think_decision");

        verticalLayout_4->addWidget(b_benchmark_think_decision);

        b_benchmark_rotate_organism = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_rotate_organism->setObjectName("b_benchmark_rotate_organism");

        verticalLayout_4->addWidget(b_benchmark_rotate_organism);

        b_benchmark_move_organism = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_move_organism->setObjectName("b_benchmark_move_organism");

        verticalLayout_4->addWidget(b_benchmark_move_organism);

        b_benchmark_try_make_child = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_try_make_child->setObjectName("b_benchmark_try_make_child");
        b_benchmark_try_make_child->setEnabled(true);

        verticalLayout_4->addWidget(b_benchmark_try_make_child);

        verticalSpacer_3 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_3);

        b_stop_benchmarks = new QPushButton(scrollAreaWidgetContents);
        b_stop_benchmarks->setObjectName("b_stop_benchmarks");

        verticalLayout_4->addWidget(b_stop_benchmarks);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName("verticalLayout_5");
        benchmarks_output_text_edit = new QTextEdit(scrollAreaWidgetContents);
        benchmarks_output_text_edit->setObjectName("benchmarks_output_text_edit");
        benchmarks_output_text_edit->setReadOnly(true);

        verticalLayout_5->addWidget(benchmarks_output_text_edit);


        horizontalLayout->addLayout(verticalLayout_5);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 1);

        verticalLayout_3->addLayout(horizontalLayout);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_2->addWidget(scrollArea);

        tabWidget->addTab(main_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName("settings_tab");
        verticalLayout_6 = new QVBoxLayout(settings_tab);
        verticalLayout_6->setObjectName("verticalLayout_6");
        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_5);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        label_2 = new QLabel(settings_tab);
        label_2->setObjectName("label_2");

        horizontalLayout_2->addWidget(label_2);

        le_grid_width = new QLineEdit(settings_tab);
        le_grid_width->setObjectName("le_grid_width");

        horizontalLayout_2->addWidget(le_grid_width);

        label = new QLabel(settings_tab);
        label->setObjectName("label");

        horizontalLayout_2->addWidget(label);

        le_grid_height = new QLineEdit(settings_tab);
        le_grid_height->setObjectName("le_grid_height");

        horizontalLayout_2->addWidget(le_grid_height);


        verticalLayout_6->addLayout(horizontalLayout_2);

        b_apply_grid_size = new QPushButton(settings_tab);
        b_apply_grid_size->setObjectName("b_apply_grid_size");

        verticalLayout_6->addWidget(b_apply_grid_size);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        label_3 = new QLabel(settings_tab);
        label_3->setObjectName("label_3");

        horizontalLayout_3->addWidget(label_3);

        le_num_benchmark_organisms = new QLineEdit(settings_tab);
        le_num_benchmark_organisms->setObjectName("le_num_benchmark_organisms");

        horizontalLayout_3->addWidget(le_num_benchmark_organisms);


        verticalLayout_6->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        label_4 = new QLabel(settings_tab);
        label_4->setObjectName("label_4");

        horizontalLayout_4->addWidget(label_4);

        le_num_iterations = new QLineEdit(settings_tab);
        le_num_iterations->setObjectName("le_num_iterations");

        horizontalLayout_4->addWidget(le_num_iterations);


        verticalLayout_6->addLayout(horizontalLayout_4);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        label_5 = new QLabel(settings_tab);
        label_5->setObjectName("label_5");

        horizontalLayout_5->addWidget(label_5);

        le_organisms_diameter = new QLineEdit(settings_tab);
        le_organisms_diameter->setObjectName("le_organisms_diameter");

        horizontalLayout_5->addWidget(le_organisms_diameter);


        verticalLayout_6->addLayout(horizontalLayout_5);

        verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_6);

        tabWidget->addTab(settings_tab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Benchmark);
        QObject::connect(b_benchmark_produce_food, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_produce_food_slot()));
        QObject::connect(b_benchmark_apply_damage, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_apply_damage_slot()));
        QObject::connect(b_benchmark_eat_food, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_eat_food_slot()));
        QObject::connect(b_benchmark_get_observations, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_get_observations_slot()));
        QObject::connect(b_benchmark_move_organism, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_move_organism_slot()));
        QObject::connect(b_benchmark_rotate_organism, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_rotate_organism_slot()));
        QObject::connect(b_benchmark_think_decision, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_think_decision_slot()));
        QObject::connect(b_benchmark_tick_lifetime, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_tick_lifetime_slot()));
        QObject::connect(b_benchmark_try_make_child, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_try_make_child_slot()));
        QObject::connect(b_run_all_benchmarks, SIGNAL(clicked()), Benchmark, SLOT(b_run_all_benchmarks_slot()));
        QObject::connect(b_stop_benchmarks, SIGNAL(clicked()), Benchmark, SLOT(b_stop_benchmarks_slot()));
        QObject::connect(le_grid_width, SIGNAL(returnPressed()), Benchmark, SLOT(le_grid_width_slot()));
        QObject::connect(le_grid_height, SIGNAL(returnPressed()), Benchmark, SLOT(le_grid_height_slot()));
        QObject::connect(b_apply_grid_size, SIGNAL(clicked()), Benchmark, SLOT(b_apply_grid_size_slot()));
        QObject::connect(le_num_benchmark_organisms, SIGNAL(returnPressed()), Benchmark, SLOT(le_num_benchmark_organisms_slot()));
        QObject::connect(le_num_iterations, SIGNAL(returnPressed()), Benchmark, SLOT(le_num_iterations_slot()));
        QObject::connect(le_organisms_diameter, SIGNAL(returnPressed()), Benchmark, SLOT(le_num_organisms_diameter_slot()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Benchmark);
    } // setupUi

    void retranslateUi(QWidget *Benchmark)
    {
        Benchmark->setWindowTitle(QCoreApplication::translate("Benchmark", "Form", nullptr));
        b_run_all_benchmarks->setText(QCoreApplication::translate("Benchmark", "Run all benchmarks", nullptr));
        b_benchmark_produce_food->setText(QCoreApplication::translate("Benchmark", "Benchmark produce food", nullptr));
        b_benchmark_eat_food->setText(QCoreApplication::translate("Benchmark", "Benchmark eat food", nullptr));
        b_benchmark_apply_damage->setText(QCoreApplication::translate("Benchmark", "Benchmark apply damage", nullptr));
        b_benchmark_tick_lifetime->setText(QCoreApplication::translate("Benchmark", "Benchmark tick lifetime", nullptr));
        b_benchmark_get_observations->setText(QCoreApplication::translate("Benchmark", "Benchmark get observations", nullptr));
        b_benchmark_think_decision->setText(QCoreApplication::translate("Benchmark", "Benchmark think decision", nullptr));
        b_benchmark_rotate_organism->setText(QCoreApplication::translate("Benchmark", "Benchmark rotate organism", nullptr));
        b_benchmark_move_organism->setText(QCoreApplication::translate("Benchmark", "Benchmark move organism", nullptr));
        b_benchmark_try_make_child->setText(QCoreApplication::translate("Benchmark", "Benchmark try make child", nullptr));
        b_stop_benchmarks->setText(QCoreApplication::translate("Benchmark", "Stop benchmarks / Finish benchmarking", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(main_tab), QCoreApplication::translate("Benchmark", "Benchmark", nullptr));
        label_2->setText(QCoreApplication::translate("Benchmark", "Grid width", nullptr));
        label->setText(QCoreApplication::translate("Benchmark", "Grid height", nullptr));
        b_apply_grid_size->setText(QCoreApplication::translate("Benchmark", "Apply grid size", nullptr));
        label_3->setText(QCoreApplication::translate("Benchmark", "Num benchmark organisms", nullptr));
        label_4->setText(QCoreApplication::translate("Benchmark", "Num iterations", nullptr));
        label_5->setText(QCoreApplication::translate("Benchmark", "Organisms diameter", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(settings_tab), QCoreApplication::translate("Benchmark", "Benchmark Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Benchmark: public Ui_Benchmark {};
} // namespace Ui

QT_END_NAMESPACE

#endif // BENCHMARKUI_H
