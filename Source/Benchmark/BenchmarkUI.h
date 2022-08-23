/********************************************************************************
** Form generated from reading UI file 'benchmark.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef BENCHMARKUI_H
#define BENCHMARKUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
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
    QPushButton *b_benchmark_erase_organisms;
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

    void setupUi(QWidget *Benchmark)
    {
        if (Benchmark->objectName().isEmpty())
            Benchmark->setObjectName(QString::fromUtf8("Benchmark"));
        Benchmark->resize(1031, 618);
        verticalLayout = new QVBoxLayout(Benchmark);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Benchmark);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        main_tab = new QWidget();
        main_tab->setObjectName(QString::fromUtf8("main_tab"));
        verticalLayout_2 = new QVBoxLayout(main_tab);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(main_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1025, 585));
        verticalLayout_3 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, -1, 0, 0);
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_2);

        b_run_all_benchmarks = new QPushButton(scrollAreaWidgetContents);
        b_run_all_benchmarks->setObjectName(QString::fromUtf8("b_run_all_benchmarks"));

        verticalLayout_4->addWidget(b_run_all_benchmarks);

        verticalSpacer_4 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_4);

        b_benchmark_produce_food = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_produce_food->setObjectName(QString::fromUtf8("b_benchmark_produce_food"));

        verticalLayout_4->addWidget(b_benchmark_produce_food);

        b_benchmark_eat_food = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_eat_food->setObjectName(QString::fromUtf8("b_benchmark_eat_food"));

        verticalLayout_4->addWidget(b_benchmark_eat_food);

        b_benchmark_apply_damage = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_apply_damage->setObjectName(QString::fromUtf8("b_benchmark_apply_damage"));

        verticalLayout_4->addWidget(b_benchmark_apply_damage);

        b_benchmark_tick_lifetime = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_tick_lifetime->setObjectName(QString::fromUtf8("b_benchmark_tick_lifetime"));

        verticalLayout_4->addWidget(b_benchmark_tick_lifetime);

        b_benchmark_erase_organisms = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_erase_organisms->setObjectName(QString::fromUtf8("b_benchmark_erase_organisms"));

        verticalLayout_4->addWidget(b_benchmark_erase_organisms);

        b_benchmark_get_observations = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_get_observations->setObjectName(QString::fromUtf8("b_benchmark_get_observations"));

        verticalLayout_4->addWidget(b_benchmark_get_observations);

        b_benchmark_think_decision = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_think_decision->setObjectName(QString::fromUtf8("b_benchmark_think_decision"));

        verticalLayout_4->addWidget(b_benchmark_think_decision);

        b_benchmark_rotate_organism = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_rotate_organism->setObjectName(QString::fromUtf8("b_benchmark_rotate_organism"));

        verticalLayout_4->addWidget(b_benchmark_rotate_organism);

        b_benchmark_move_organism = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_move_organism->setObjectName(QString::fromUtf8("b_benchmark_move_organism"));

        verticalLayout_4->addWidget(b_benchmark_move_organism);

        b_benchmark_try_make_child = new QPushButton(scrollAreaWidgetContents);
        b_benchmark_try_make_child->setObjectName(QString::fromUtf8("b_benchmark_try_make_child"));
        b_benchmark_try_make_child->setEnabled(false);

        verticalLayout_4->addWidget(b_benchmark_try_make_child);

        verticalSpacer_3 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_3);

        b_stop_benchmarks = new QPushButton(scrollAreaWidgetContents);
        b_stop_benchmarks->setObjectName(QString::fromUtf8("b_stop_benchmarks"));

        verticalLayout_4->addWidget(b_stop_benchmarks);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        benchmarks_output_text_edit = new QTextEdit(scrollAreaWidgetContents);
        benchmarks_output_text_edit->setObjectName(QString::fromUtf8("benchmarks_output_text_edit"));
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
        settings_tab->setObjectName(QString::fromUtf8("settings_tab"));
        tabWidget->addTab(settings_tab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Benchmark);
        QObject::connect(b_benchmark_produce_food, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_produce_food_slot()));
        QObject::connect(b_benchmark_apply_damage, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_apply_damage_slot()));
        QObject::connect(b_benchmark_eat_food, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_eat_food_slot()));
        QObject::connect(b_benchmark_erase_organisms, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_erase_organisms_slot()));
        QObject::connect(b_benchmark_get_observations, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_get_observations_slot()));
        QObject::connect(b_benchmark_move_organism, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_move_organism_slot()));
        QObject::connect(b_benchmark_rotate_organism, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_rotate_organism_slot()));
        QObject::connect(b_benchmark_think_decision, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_think_decision_slot()));
        QObject::connect(b_benchmark_tick_lifetime, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_tick_lifetime_slot()));
        QObject::connect(b_benchmark_try_make_child, SIGNAL(clicked()), Benchmark, SLOT(b_benchmark_try_make_child_slot()));
        QObject::connect(b_run_all_benchmarks, SIGNAL(clicked()), Benchmark, SLOT(b_run_all_benchmarks_slot()));
        QObject::connect(b_stop_benchmarks, SIGNAL(clicked()), Benchmark, SLOT(b_stop_benchmarks_slot()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Benchmark);
    } // setupUi

    void retranslateUi(QWidget *Benchmark)
    {
        Benchmark->setWindowTitle(QApplication::translate("Benchmark", "Form", nullptr));
        b_run_all_benchmarks->setText(QApplication::translate("Benchmark", "Run all benchmarks", nullptr));
        b_benchmark_produce_food->setText(QApplication::translate("Benchmark", "Benchmark produce food", nullptr));
        b_benchmark_eat_food->setText(QApplication::translate("Benchmark", "Benchmark eat food", nullptr));
        b_benchmark_apply_damage->setText(QApplication::translate("Benchmark", "Benchmark apply damage", nullptr));
        b_benchmark_tick_lifetime->setText(QApplication::translate("Benchmark", "Benchmark tick lifetime", nullptr));
        b_benchmark_erase_organisms->setText(QApplication::translate("Benchmark", "Benchmark erase organisms", nullptr));
        b_benchmark_get_observations->setText(QApplication::translate("Benchmark", "Benchmark get observations", nullptr));
        b_benchmark_think_decision->setText(QApplication::translate("Benchmark", "Benchmark think decision", nullptr));
        b_benchmark_rotate_organism->setText(QApplication::translate("Benchmark", "Benchmark rotate organism_index", nullptr));
        b_benchmark_move_organism->setText(QApplication::translate("Benchmark", "Benchmark move organism_index", nullptr));
        b_benchmark_try_make_child->setText(QApplication::translate("Benchmark", "Benchmark try make child", nullptr));
        b_stop_benchmarks->setText(QApplication::translate("Benchmark", "Stop benchmarks / Finish benchmarking", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(main_tab), QApplication::translate("Benchmark", "Benchmark", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(settings_tab), QApplication::translate("Benchmark", "Benchmark Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Benchmark: public Ui_Benchmark {};
} // namespace Ui

QT_END_NAMESPACE

#endif // BENCHMARKUI_H
