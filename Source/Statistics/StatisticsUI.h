/********************************************************************************
** Form generated from reading UI file 'statistics.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef STATISTICSUI_H
#define STATISTICSUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Statistics
{
public:
    QVBoxLayout *verticalLayout;
    QScrollArea *scrollArea_4;
    QWidget *scrollAreaWidgetContents_4;
    QVBoxLayout *verticalLayout_24;
    QWidget *widget_5;
    QVBoxLayout *verticalLayout_37;
    QLabel *lb_simulation_size;
    QLabel *lb_organisms_memory_consumption;
    QLabel *lb_total_engine_ticks;
    QLabel *lb_child_organisms;
    QLabel *lb_child_organisms_in_use;
    QLabel *lb_child_organisms_capacity;
    QLabel *lb_total_organisms;
    QLabel *lb_dead_organisms;
    QLabel *lb_last_alive_position;
    QLabel *lb_dead_inside;
    QLabel *lb_dead_outside;
    QLabel *lb_organisms_capacity;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout_47;
    QVBoxLayout *verticalLayout_38;
    QLabel *lb_moving_organisms;
    QLabel *lb_organisms_with_eyes;
    QLabel *lb_avg_org_lifetime_2;
    QLabel *lb_avg_age_2;
    QLabel *lb_average_moving_range;
    QLabel *lb_organism_size_2;
    QLabel *lb_anatomy_mutation_rate_2;
    QLabel *lb_brain_mutation_rate_2;
    QLabel *lb_mouth_num_2;
    QLabel *lb_producer_num_2;
    QLabel *lb_mover_num_2;
    QLabel *lb_killer_num_2;
    QLabel *lb_armor_num_2;
    QLabel *lb_eye_num_2;
    QVBoxLayout *verticalLayout_40;
    QLabel *lb_stationary_organisms;
    QLabel *lb_organism_size_3;
    QLabel *lb_avg_org_lifetime_3;
    QLabel *lb_avg_age_3;
    QLabel *lb_anatomy_mutation_rate_3;
    QLabel *lb_brain_mutation_rate_3;
    QLabel *lb_producer_num_3;
    QLabel *lb_mouth_num_3;
    QLabel *lb_killer_num_3;
    QLabel *lb_armor_num_3;
    QLabel *lb_eye_num_3;
    QVBoxLayout *verticalLayout_10;
    QLabel *lb_organisms_alive_2;
    QLabel *lb_organism_size_4;
    QLabel *lb_avg_org_lifetime_4;
    QLabel *lb_avg_age_4;
    QLabel *lb_anatomy_mutation_rate_4;
    QLabel *lb_brain_mutation_rate_4;
    QLabel *lb_producer_num_4;
    QLabel *lb_mover_num_4;
    QLabel *lb_mouth_num_4;
    QLabel *lb_killer_num_4;
    QLabel *lb_armor_num_4;
    QLabel *lb_eye_num_4;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *Statistics)
    {
        if (Statistics->objectName().isEmpty())
            Statistics->setObjectName(QString::fromUtf8("Statistics"));
        Statistics->resize(725, 432);
        verticalLayout = new QVBoxLayout(Statistics);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea_4 = new QScrollArea(Statistics);
        scrollArea_4->setObjectName(QString::fromUtf8("scrollArea_4"));
        scrollArea_4->setMinimumSize(QSize(0, 0));
        scrollArea_4->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QString::fromUtf8("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 709, 636));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(scrollAreaWidgetContents_4->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_4->setSizePolicy(sizePolicy);
        verticalLayout_24 = new QVBoxLayout(scrollAreaWidgetContents_4);
        verticalLayout_24->setObjectName(QString::fromUtf8("verticalLayout_24"));
        verticalLayout_24->setContentsMargins(9, -1, -1, -1);
        widget_5 = new QWidget(scrollAreaWidgetContents_4);
        widget_5->setObjectName(QString::fromUtf8("widget_5"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(widget_5->sizePolicy().hasHeightForWidth());
        widget_5->setSizePolicy(sizePolicy1);
        widget_5->setMinimumSize(QSize(0, 0));
        verticalLayout_37 = new QVBoxLayout(widget_5);
        verticalLayout_37->setObjectName(QString::fromUtf8("verticalLayout_37"));
        verticalLayout_37->setContentsMargins(0, 0, 0, 0);
        lb_simulation_size = new QLabel(widget_5);
        lb_simulation_size->setObjectName(QString::fromUtf8("lb_simulation_size"));

        verticalLayout_37->addWidget(lb_simulation_size);

        lb_organisms_memory_consumption = new QLabel(widget_5);
        lb_organisms_memory_consumption->setObjectName(QString::fromUtf8("lb_organisms_memory_consumption"));

        verticalLayout_37->addWidget(lb_organisms_memory_consumption);

        lb_total_engine_ticks = new QLabel(widget_5);
        lb_total_engine_ticks->setObjectName(QString::fromUtf8("lb_total_engine_ticks"));

        verticalLayout_37->addWidget(lb_total_engine_ticks);

        lb_child_organisms = new QLabel(widget_5);
        lb_child_organisms->setObjectName(QString::fromUtf8("lb_child_organisms"));

        verticalLayout_37->addWidget(lb_child_organisms);

        lb_child_organisms_in_use = new QLabel(widget_5);
        lb_child_organisms_in_use->setObjectName(QString::fromUtf8("lb_child_organisms_in_use"));

        verticalLayout_37->addWidget(lb_child_organisms_in_use);

        lb_child_organisms_capacity = new QLabel(widget_5);
        lb_child_organisms_capacity->setObjectName(QString::fromUtf8("lb_child_organisms_capacity"));

        verticalLayout_37->addWidget(lb_child_organisms_capacity);

        lb_total_organisms = new QLabel(widget_5);
        lb_total_organisms->setObjectName(QString::fromUtf8("lb_total_organisms"));

        verticalLayout_37->addWidget(lb_total_organisms);

        lb_dead_organisms = new QLabel(widget_5);
        lb_dead_organisms->setObjectName(QString::fromUtf8("lb_dead_organisms"));

        verticalLayout_37->addWidget(lb_dead_organisms);

        lb_last_alive_position = new QLabel(widget_5);
        lb_last_alive_position->setObjectName(QString::fromUtf8("lb_last_alive_position"));

        verticalLayout_37->addWidget(lb_last_alive_position);

        lb_dead_inside = new QLabel(widget_5);
        lb_dead_inside->setObjectName(QString::fromUtf8("lb_dead_inside"));

        verticalLayout_37->addWidget(lb_dead_inside);

        lb_dead_outside = new QLabel(widget_5);
        lb_dead_outside->setObjectName(QString::fromUtf8("lb_dead_outside"));

        verticalLayout_37->addWidget(lb_dead_outside);

        lb_organisms_capacity = new QLabel(widget_5);
        lb_organisms_capacity->setObjectName(QString::fromUtf8("lb_organisms_capacity"));

        verticalLayout_37->addWidget(lb_organisms_capacity);

        verticalSpacer = new QSpacerItem(20, 10, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_37->addItem(verticalSpacer);

        horizontalLayout_47 = new QHBoxLayout();
        horizontalLayout_47->setObjectName(QString::fromUtf8("horizontalLayout_47"));
        verticalLayout_38 = new QVBoxLayout();
        verticalLayout_38->setObjectName(QString::fromUtf8("verticalLayout_38"));
        lb_moving_organisms = new QLabel(widget_5);
        lb_moving_organisms->setObjectName(QString::fromUtf8("lb_moving_organisms"));

        verticalLayout_38->addWidget(lb_moving_organisms);

        lb_organisms_with_eyes = new QLabel(widget_5);
        lb_organisms_with_eyes->setObjectName(QString::fromUtf8("lb_organisms_with_eyes"));

        verticalLayout_38->addWidget(lb_organisms_with_eyes);

        lb_avg_org_lifetime_2 = new QLabel(widget_5);
        lb_avg_org_lifetime_2->setObjectName(QString::fromUtf8("lb_avg_org_lifetime_2"));

        verticalLayout_38->addWidget(lb_avg_org_lifetime_2);

        lb_avg_age_2 = new QLabel(widget_5);
        lb_avg_age_2->setObjectName(QString::fromUtf8("lb_avg_age_2"));

        verticalLayout_38->addWidget(lb_avg_age_2);

        lb_average_moving_range = new QLabel(widget_5);
        lb_average_moving_range->setObjectName(QString::fromUtf8("lb_average_moving_range"));

        verticalLayout_38->addWidget(lb_average_moving_range);

        lb_organism_size_2 = new QLabel(widget_5);
        lb_organism_size_2->setObjectName(QString::fromUtf8("lb_organism_size_2"));

        verticalLayout_38->addWidget(lb_organism_size_2);

        lb_anatomy_mutation_rate_2 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_2->setObjectName(QString::fromUtf8("lb_anatomy_mutation_rate_2"));

        verticalLayout_38->addWidget(lb_anatomy_mutation_rate_2);

        lb_brain_mutation_rate_2 = new QLabel(widget_5);
        lb_brain_mutation_rate_2->setObjectName(QString::fromUtf8("lb_brain_mutation_rate_2"));

        verticalLayout_38->addWidget(lb_brain_mutation_rate_2);

        lb_mouth_num_2 = new QLabel(widget_5);
        lb_mouth_num_2->setObjectName(QString::fromUtf8("lb_mouth_num_2"));

        verticalLayout_38->addWidget(lb_mouth_num_2);

        lb_producer_num_2 = new QLabel(widget_5);
        lb_producer_num_2->setObjectName(QString::fromUtf8("lb_producer_num_2"));

        verticalLayout_38->addWidget(lb_producer_num_2);

        lb_mover_num_2 = new QLabel(widget_5);
        lb_mover_num_2->setObjectName(QString::fromUtf8("lb_mover_num_2"));

        verticalLayout_38->addWidget(lb_mover_num_2);

        lb_killer_num_2 = new QLabel(widget_5);
        lb_killer_num_2->setObjectName(QString::fromUtf8("lb_killer_num_2"));

        verticalLayout_38->addWidget(lb_killer_num_2);

        lb_armor_num_2 = new QLabel(widget_5);
        lb_armor_num_2->setObjectName(QString::fromUtf8("lb_armor_num_2"));

        verticalLayout_38->addWidget(lb_armor_num_2);

        lb_eye_num_2 = new QLabel(widget_5);
        lb_eye_num_2->setObjectName(QString::fromUtf8("lb_eye_num_2"));

        verticalLayout_38->addWidget(lb_eye_num_2);


        horizontalLayout_47->addLayout(verticalLayout_38);

        verticalLayout_40 = new QVBoxLayout();
        verticalLayout_40->setObjectName(QString::fromUtf8("verticalLayout_40"));
        lb_stationary_organisms = new QLabel(widget_5);
        lb_stationary_organisms->setObjectName(QString::fromUtf8("lb_stationary_organisms"));

        verticalLayout_40->addWidget(lb_stationary_organisms);

        lb_organism_size_3 = new QLabel(widget_5);
        lb_organism_size_3->setObjectName(QString::fromUtf8("lb_organism_size_3"));

        verticalLayout_40->addWidget(lb_organism_size_3);

        lb_avg_org_lifetime_3 = new QLabel(widget_5);
        lb_avg_org_lifetime_3->setObjectName(QString::fromUtf8("lb_avg_org_lifetime_3"));

        verticalLayout_40->addWidget(lb_avg_org_lifetime_3);

        lb_avg_age_3 = new QLabel(widget_5);
        lb_avg_age_3->setObjectName(QString::fromUtf8("lb_avg_age_3"));

        verticalLayout_40->addWidget(lb_avg_age_3);

        lb_anatomy_mutation_rate_3 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_3->setObjectName(QString::fromUtf8("lb_anatomy_mutation_rate_3"));

        verticalLayout_40->addWidget(lb_anatomy_mutation_rate_3);

        lb_brain_mutation_rate_3 = new QLabel(widget_5);
        lb_brain_mutation_rate_3->setObjectName(QString::fromUtf8("lb_brain_mutation_rate_3"));

        verticalLayout_40->addWidget(lb_brain_mutation_rate_3);

        lb_producer_num_3 = new QLabel(widget_5);
        lb_producer_num_3->setObjectName(QString::fromUtf8("lb_producer_num_3"));

        verticalLayout_40->addWidget(lb_producer_num_3);

        lb_mouth_num_3 = new QLabel(widget_5);
        lb_mouth_num_3->setObjectName(QString::fromUtf8("lb_mouth_num_3"));

        verticalLayout_40->addWidget(lb_mouth_num_3);

        lb_killer_num_3 = new QLabel(widget_5);
        lb_killer_num_3->setObjectName(QString::fromUtf8("lb_killer_num_3"));

        verticalLayout_40->addWidget(lb_killer_num_3);

        lb_armor_num_3 = new QLabel(widget_5);
        lb_armor_num_3->setObjectName(QString::fromUtf8("lb_armor_num_3"));

        verticalLayout_40->addWidget(lb_armor_num_3);

        lb_eye_num_3 = new QLabel(widget_5);
        lb_eye_num_3->setObjectName(QString::fromUtf8("lb_eye_num_3"));

        verticalLayout_40->addWidget(lb_eye_num_3);


        horizontalLayout_47->addLayout(verticalLayout_40);

        verticalLayout_10 = new QVBoxLayout();
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        lb_organisms_alive_2 = new QLabel(widget_5);
        lb_organisms_alive_2->setObjectName(QString::fromUtf8("lb_organisms_alive_2"));

        verticalLayout_10->addWidget(lb_organisms_alive_2);

        lb_organism_size_4 = new QLabel(widget_5);
        lb_organism_size_4->setObjectName(QString::fromUtf8("lb_organism_size_4"));

        verticalLayout_10->addWidget(lb_organism_size_4);

        lb_avg_org_lifetime_4 = new QLabel(widget_5);
        lb_avg_org_lifetime_4->setObjectName(QString::fromUtf8("lb_avg_org_lifetime_4"));

        verticalLayout_10->addWidget(lb_avg_org_lifetime_4);

        lb_avg_age_4 = new QLabel(widget_5);
        lb_avg_age_4->setObjectName(QString::fromUtf8("lb_avg_age_4"));

        verticalLayout_10->addWidget(lb_avg_age_4);

        lb_anatomy_mutation_rate_4 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_4->setObjectName(QString::fromUtf8("lb_anatomy_mutation_rate_4"));

        verticalLayout_10->addWidget(lb_anatomy_mutation_rate_4);

        lb_brain_mutation_rate_4 = new QLabel(widget_5);
        lb_brain_mutation_rate_4->setObjectName(QString::fromUtf8("lb_brain_mutation_rate_4"));

        verticalLayout_10->addWidget(lb_brain_mutation_rate_4);

        lb_producer_num_4 = new QLabel(widget_5);
        lb_producer_num_4->setObjectName(QString::fromUtf8("lb_producer_num_4"));

        verticalLayout_10->addWidget(lb_producer_num_4);

        lb_mover_num_4 = new QLabel(widget_5);
        lb_mover_num_4->setObjectName(QString::fromUtf8("lb_mover_num_4"));

        verticalLayout_10->addWidget(lb_mover_num_4);

        lb_mouth_num_4 = new QLabel(widget_5);
        lb_mouth_num_4->setObjectName(QString::fromUtf8("lb_mouth_num_4"));

        verticalLayout_10->addWidget(lb_mouth_num_4);

        lb_killer_num_4 = new QLabel(widget_5);
        lb_killer_num_4->setObjectName(QString::fromUtf8("lb_killer_num_4"));

        verticalLayout_10->addWidget(lb_killer_num_4);

        lb_armor_num_4 = new QLabel(widget_5);
        lb_armor_num_4->setObjectName(QString::fromUtf8("lb_armor_num_4"));

        verticalLayout_10->addWidget(lb_armor_num_4);

        lb_eye_num_4 = new QLabel(widget_5);
        lb_eye_num_4->setObjectName(QString::fromUtf8("lb_eye_num_4"));

        verticalLayout_10->addWidget(lb_eye_num_4);


        horizontalLayout_47->addLayout(verticalLayout_10);


        verticalLayout_37->addLayout(horizontalLayout_47);


        verticalLayout_24->addWidget(widget_5);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_24->addItem(verticalSpacer_2);

        scrollArea_4->setWidget(scrollAreaWidgetContents_4);

        verticalLayout->addWidget(scrollArea_4);


        retranslateUi(Statistics);

        QMetaObject::connectSlotsByName(Statistics);
    } // setupUi

    void retranslateUi(QWidget *Statistics)
    {
        Statistics->setWindowTitle(QApplication::translate("Statistics", "Statistics", nullptr));
        lb_simulation_size->setText(QApplication::translate("Statistics", "Simulation size:", nullptr));
        lb_organisms_memory_consumption->setText(QApplication::translate("Statistics", "Organism's memory consumption:", nullptr));
        lb_total_engine_ticks->setText(QApplication::translate("Statistics", "Total engine ticks: ", nullptr));
        lb_child_organisms->setText(QApplication::translate("Statistics", "Child organisms:", nullptr));
        lb_child_organisms_in_use->setText(QApplication::translate("Statistics", "Child organisms in use:", nullptr));
        lb_child_organisms_capacity->setText(QApplication::translate("Statistics", "Child organisms capacity:", nullptr));
        lb_total_organisms->setText(QApplication::translate("Statistics", "Total organisms:", nullptr));
        lb_dead_organisms->setText(QApplication::translate("Statistics", "Dead organisms:", nullptr));
        lb_last_alive_position->setText(QApplication::translate("Statistics", "Last alive position:", nullptr));
        lb_dead_inside->setText(QApplication::translate("Statistics", "Dead inside:", nullptr));
        lb_dead_outside->setText(QApplication::translate("Statistics", "Dead outside:", nullptr));
        lb_organisms_capacity->setText(QApplication::translate("Statistics", "Organisms capacity:", nullptr));
        lb_moving_organisms->setText(QApplication::translate("Statistics", "Moving organisms:", nullptr));
        lb_organisms_with_eyes->setText(QApplication::translate("Statistics", "Organisms with eyes:", nullptr));
        lb_avg_org_lifetime_2->setText(QApplication::translate("Statistics", "Avg organism lifetime:", nullptr));
        lb_avg_age_2->setText(QApplication::translate("Statistics", "Avg organism age:", nullptr));
        lb_average_moving_range->setText(QApplication::translate("Statistics", "Avg moving range:", nullptr));
        lb_organism_size_2->setText(QApplication::translate("Statistics", "Avg organism size:", nullptr));
        lb_anatomy_mutation_rate_2->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", nullptr));
        lb_brain_mutation_rate_2->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", nullptr));
        lb_mouth_num_2->setText(QApplication::translate("Statistics", "Avg mouth num: ", nullptr));
        lb_producer_num_2->setText(QApplication::translate("Statistics", "Avg producer num: ", nullptr));
        lb_mover_num_2->setText(QApplication::translate("Statistics", "Avg mover num:", nullptr));
        lb_killer_num_2->setText(QApplication::translate("Statistics", "Avg killer num:", nullptr));
        lb_armor_num_2->setText(QApplication::translate("Statistics", "Avg armor num: ", nullptr));
        lb_eye_num_2->setText(QApplication::translate("Statistics", "Avg eye num: ", nullptr));
        lb_stationary_organisms->setText(QApplication::translate("Statistics", "Stationary organisms:", nullptr));
        lb_organism_size_3->setText(QApplication::translate("Statistics", "Avg organism size:", nullptr));
        lb_avg_org_lifetime_3->setText(QApplication::translate("Statistics", "Avg organism lifetime:", nullptr));
        lb_avg_age_3->setText(QApplication::translate("Statistics", "Avg organism age:", nullptr));
        lb_anatomy_mutation_rate_3->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", nullptr));
        lb_brain_mutation_rate_3->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", nullptr));
        lb_producer_num_3->setText(QApplication::translate("Statistics", "Avg producer num: ", nullptr));
        lb_mouth_num_3->setText(QApplication::translate("Statistics", "Avg mouth num: ", nullptr));
        lb_killer_num_3->setText(QApplication::translate("Statistics", "Avg killer num:", nullptr));
        lb_armor_num_3->setText(QApplication::translate("Statistics", "Avg armor num: ", nullptr));
        lb_eye_num_3->setText(QApplication::translate("Statistics", "Avg eye num: ", nullptr));
        lb_organisms_alive_2->setText(QApplication::translate("Statistics", "Organisms alive:", nullptr));
        lb_organism_size_4->setText(QApplication::translate("Statistics", "Avg organism size:", nullptr));
        lb_avg_org_lifetime_4->setText(QApplication::translate("Statistics", "Avg organism lifetime:", nullptr));
        lb_avg_age_4->setText(QApplication::translate("Statistics", "Avg organism age:", nullptr));
        lb_anatomy_mutation_rate_4->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", nullptr));
        lb_brain_mutation_rate_4->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", nullptr));
        lb_producer_num_4->setText(QApplication::translate("Statistics", "Avg producer num: ", nullptr));
        lb_mover_num_4->setText(QApplication::translate("Statistics", "Avg mover num:", nullptr));
        lb_mouth_num_4->setText(QApplication::translate("Statistics", "Avg mouth num: ", nullptr));
        lb_killer_num_4->setText(QApplication::translate("Statistics", "Avg killer num:", nullptr));
        lb_armor_num_4->setText(QApplication::translate("Statistics", "Avg armor num: ", nullptr));
        lb_eye_num_4->setText(QApplication::translate("Statistics", "Avg eye num: ", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Statistics: public Ui_Statistics {};
} // namespace Ui

QT_END_NAMESPACE

#endif // STATISTICSUI_H
