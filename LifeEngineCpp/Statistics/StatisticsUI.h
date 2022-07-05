/********************************************************************************
** Form generated from reading UI file 'statistics.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef STATISTICSUI_H
#define STATISTICSUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
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
            Statistics->setObjectName(QStringLiteral("Statistics"));
        Statistics->resize(725, 432);
        verticalLayout = new QVBoxLayout(Statistics);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea_4 = new QScrollArea(Statistics);
        scrollArea_4->setObjectName(QStringLiteral("scrollArea_4"));
        scrollArea_4->setMinimumSize(QSize(0, 0));
        scrollArea_4->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QStringLiteral("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 723, 430));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(scrollAreaWidgetContents_4->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_4->setSizePolicy(sizePolicy);
        verticalLayout_24 = new QVBoxLayout(scrollAreaWidgetContents_4);
        verticalLayout_24->setObjectName(QStringLiteral("verticalLayout_24"));
        verticalLayout_24->setContentsMargins(9, -1, -1, -1);
        widget_5 = new QWidget(scrollAreaWidgetContents_4);
        widget_5->setObjectName(QStringLiteral("widget_5"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(widget_5->sizePolicy().hasHeightForWidth());
        widget_5->setSizePolicy(sizePolicy1);
        widget_5->setMinimumSize(QSize(0, 0));
        verticalLayout_37 = new QVBoxLayout(widget_5);
        verticalLayout_37->setObjectName(QStringLiteral("verticalLayout_37"));
        verticalLayout_37->setContentsMargins(0, 0, 0, 0);
        lb_simulation_size = new QLabel(widget_5);
        lb_simulation_size->setObjectName(QStringLiteral("lb_simulation_size"));

        verticalLayout_37->addWidget(lb_simulation_size);

        lb_organisms_memory_consumption = new QLabel(widget_5);
        lb_organisms_memory_consumption->setObjectName(QStringLiteral("lb_organisms_memory_consumption"));

        verticalLayout_37->addWidget(lb_organisms_memory_consumption);

        lb_total_engine_ticks = new QLabel(widget_5);
        lb_total_engine_ticks->setObjectName(QStringLiteral("lb_total_engine_ticks"));

        verticalLayout_37->addWidget(lb_total_engine_ticks);

        verticalSpacer = new QSpacerItem(20, 10, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_37->addItem(verticalSpacer);

        horizontalLayout_47 = new QHBoxLayout();
        horizontalLayout_47->setObjectName(QStringLiteral("horizontalLayout_47"));
        verticalLayout_38 = new QVBoxLayout();
        verticalLayout_38->setObjectName(QStringLiteral("verticalLayout_38"));
        lb_moving_organisms = new QLabel(widget_5);
        lb_moving_organisms->setObjectName(QStringLiteral("lb_moving_organisms"));

        verticalLayout_38->addWidget(lb_moving_organisms);

        lb_organisms_with_eyes = new QLabel(widget_5);
        lb_organisms_with_eyes->setObjectName(QStringLiteral("lb_organisms_with_eyes"));

        verticalLayout_38->addWidget(lb_organisms_with_eyes);

        lb_avg_org_lifetime_2 = new QLabel(widget_5);
        lb_avg_org_lifetime_2->setObjectName(QStringLiteral("lb_avg_org_lifetime_2"));

        verticalLayout_38->addWidget(lb_avg_org_lifetime_2);

        lb_avg_age_2 = new QLabel(widget_5);
        lb_avg_age_2->setObjectName(QStringLiteral("lb_avg_age_2"));

        verticalLayout_38->addWidget(lb_avg_age_2);

        lb_average_moving_range = new QLabel(widget_5);
        lb_average_moving_range->setObjectName(QStringLiteral("lb_average_moving_range"));

        verticalLayout_38->addWidget(lb_average_moving_range);

        lb_organism_size_2 = new QLabel(widget_5);
        lb_organism_size_2->setObjectName(QStringLiteral("lb_organism_size_2"));

        verticalLayout_38->addWidget(lb_organism_size_2);

        lb_anatomy_mutation_rate_2 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_2->setObjectName(QStringLiteral("lb_anatomy_mutation_rate_2"));

        verticalLayout_38->addWidget(lb_anatomy_mutation_rate_2);

        lb_brain_mutation_rate_2 = new QLabel(widget_5);
        lb_brain_mutation_rate_2->setObjectName(QStringLiteral("lb_brain_mutation_rate_2"));

        verticalLayout_38->addWidget(lb_brain_mutation_rate_2);

        lb_mouth_num_2 = new QLabel(widget_5);
        lb_mouth_num_2->setObjectName(QStringLiteral("lb_mouth_num_2"));

        verticalLayout_38->addWidget(lb_mouth_num_2);

        lb_producer_num_2 = new QLabel(widget_5);
        lb_producer_num_2->setObjectName(QStringLiteral("lb_producer_num_2"));

        verticalLayout_38->addWidget(lb_producer_num_2);

        lb_mover_num_2 = new QLabel(widget_5);
        lb_mover_num_2->setObjectName(QStringLiteral("lb_mover_num_2"));

        verticalLayout_38->addWidget(lb_mover_num_2);

        lb_killer_num_2 = new QLabel(widget_5);
        lb_killer_num_2->setObjectName(QStringLiteral("lb_killer_num_2"));

        verticalLayout_38->addWidget(lb_killer_num_2);

        lb_armor_num_2 = new QLabel(widget_5);
        lb_armor_num_2->setObjectName(QStringLiteral("lb_armor_num_2"));

        verticalLayout_38->addWidget(lb_armor_num_2);

        lb_eye_num_2 = new QLabel(widget_5);
        lb_eye_num_2->setObjectName(QStringLiteral("lb_eye_num_2"));

        verticalLayout_38->addWidget(lb_eye_num_2);


        horizontalLayout_47->addLayout(verticalLayout_38);

        verticalLayout_40 = new QVBoxLayout();
        verticalLayout_40->setObjectName(QStringLiteral("verticalLayout_40"));
        lb_stationary_organisms = new QLabel(widget_5);
        lb_stationary_organisms->setObjectName(QStringLiteral("lb_stationary_organisms"));

        verticalLayout_40->addWidget(lb_stationary_organisms);

        lb_organism_size_3 = new QLabel(widget_5);
        lb_organism_size_3->setObjectName(QStringLiteral("lb_organism_size_3"));

        verticalLayout_40->addWidget(lb_organism_size_3);

        lb_avg_org_lifetime_3 = new QLabel(widget_5);
        lb_avg_org_lifetime_3->setObjectName(QStringLiteral("lb_avg_org_lifetime_3"));

        verticalLayout_40->addWidget(lb_avg_org_lifetime_3);

        lb_avg_age_3 = new QLabel(widget_5);
        lb_avg_age_3->setObjectName(QStringLiteral("lb_avg_age_3"));

        verticalLayout_40->addWidget(lb_avg_age_3);

        lb_anatomy_mutation_rate_3 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_3->setObjectName(QStringLiteral("lb_anatomy_mutation_rate_3"));

        verticalLayout_40->addWidget(lb_anatomy_mutation_rate_3);

        lb_brain_mutation_rate_3 = new QLabel(widget_5);
        lb_brain_mutation_rate_3->setObjectName(QStringLiteral("lb_brain_mutation_rate_3"));

        verticalLayout_40->addWidget(lb_brain_mutation_rate_3);

        lb_producer_num_3 = new QLabel(widget_5);
        lb_producer_num_3->setObjectName(QStringLiteral("lb_producer_num_3"));

        verticalLayout_40->addWidget(lb_producer_num_3);

        lb_mouth_num_3 = new QLabel(widget_5);
        lb_mouth_num_3->setObjectName(QStringLiteral("lb_mouth_num_3"));

        verticalLayout_40->addWidget(lb_mouth_num_3);

        lb_killer_num_3 = new QLabel(widget_5);
        lb_killer_num_3->setObjectName(QStringLiteral("lb_killer_num_3"));

        verticalLayout_40->addWidget(lb_killer_num_3);

        lb_armor_num_3 = new QLabel(widget_5);
        lb_armor_num_3->setObjectName(QStringLiteral("lb_armor_num_3"));

        verticalLayout_40->addWidget(lb_armor_num_3);

        lb_eye_num_3 = new QLabel(widget_5);
        lb_eye_num_3->setObjectName(QStringLiteral("lb_eye_num_3"));

        verticalLayout_40->addWidget(lb_eye_num_3);


        horizontalLayout_47->addLayout(verticalLayout_40);

        verticalLayout_10 = new QVBoxLayout();
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        lb_organisms_alive_2 = new QLabel(widget_5);
        lb_organisms_alive_2->setObjectName(QStringLiteral("lb_organisms_alive_2"));

        verticalLayout_10->addWidget(lb_organisms_alive_2);

        lb_organism_size_4 = new QLabel(widget_5);
        lb_organism_size_4->setObjectName(QStringLiteral("lb_organism_size_4"));

        verticalLayout_10->addWidget(lb_organism_size_4);

        lb_avg_org_lifetime_4 = new QLabel(widget_5);
        lb_avg_org_lifetime_4->setObjectName(QStringLiteral("lb_avg_org_lifetime_4"));

        verticalLayout_10->addWidget(lb_avg_org_lifetime_4);

        lb_avg_age_4 = new QLabel(widget_5);
        lb_avg_age_4->setObjectName(QStringLiteral("lb_avg_age_4"));

        verticalLayout_10->addWidget(lb_avg_age_4);

        lb_anatomy_mutation_rate_4 = new QLabel(widget_5);
        lb_anatomy_mutation_rate_4->setObjectName(QStringLiteral("lb_anatomy_mutation_rate_4"));

        verticalLayout_10->addWidget(lb_anatomy_mutation_rate_4);

        lb_brain_mutation_rate_4 = new QLabel(widget_5);
        lb_brain_mutation_rate_4->setObjectName(QStringLiteral("lb_brain_mutation_rate_4"));

        verticalLayout_10->addWidget(lb_brain_mutation_rate_4);

        lb_producer_num_4 = new QLabel(widget_5);
        lb_producer_num_4->setObjectName(QStringLiteral("lb_producer_num_4"));

        verticalLayout_10->addWidget(lb_producer_num_4);

        lb_mover_num_4 = new QLabel(widget_5);
        lb_mover_num_4->setObjectName(QStringLiteral("lb_mover_num_4"));

        verticalLayout_10->addWidget(lb_mover_num_4);

        lb_mouth_num_4 = new QLabel(widget_5);
        lb_mouth_num_4->setObjectName(QStringLiteral("lb_mouth_num_4"));

        verticalLayout_10->addWidget(lb_mouth_num_4);

        lb_killer_num_4 = new QLabel(widget_5);
        lb_killer_num_4->setObjectName(QStringLiteral("lb_killer_num_4"));

        verticalLayout_10->addWidget(lb_killer_num_4);

        lb_armor_num_4 = new QLabel(widget_5);
        lb_armor_num_4->setObjectName(QStringLiteral("lb_armor_num_4"));

        verticalLayout_10->addWidget(lb_armor_num_4);

        lb_eye_num_4 = new QLabel(widget_5);
        lb_eye_num_4->setObjectName(QStringLiteral("lb_eye_num_4"));

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
        Statistics->setWindowTitle(QApplication::translate("Statistics", "Statistics", Q_NULLPTR));
        lb_simulation_size->setText(QApplication::translate("Statistics", "Simulation size:", Q_NULLPTR));
        lb_organisms_memory_consumption->setText(QApplication::translate("Statistics", "Organism's memory consumption:", Q_NULLPTR));
        lb_total_engine_ticks->setText(QApplication::translate("Statistics", "Total engine ticks: ", Q_NULLPTR));
        lb_moving_organisms->setText(QApplication::translate("Statistics", "Moving organisms:", Q_NULLPTR));
        lb_organisms_with_eyes->setText(QApplication::translate("Statistics", "Organisms with eyes:", Q_NULLPTR));
        lb_avg_org_lifetime_2->setText(QApplication::translate("Statistics", "Avg organism lifetime:", Q_NULLPTR));
        lb_avg_age_2->setText(QApplication::translate("Statistics", "Avg organism age:", Q_NULLPTR));
        lb_average_moving_range->setText(QApplication::translate("Statistics", "Avg moving range:", Q_NULLPTR));
        lb_organism_size_2->setText(QApplication::translate("Statistics", "Avg organism size:", Q_NULLPTR));
        lb_anatomy_mutation_rate_2->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", Q_NULLPTR));
        lb_brain_mutation_rate_2->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", Q_NULLPTR));
        lb_mouth_num_2->setText(QApplication::translate("Statistics", "Avg mouth num: ", Q_NULLPTR));
        lb_producer_num_2->setText(QApplication::translate("Statistics", "Avg producer num: ", Q_NULLPTR));
        lb_mover_num_2->setText(QApplication::translate("Statistics", "Avg mover num:", Q_NULLPTR));
        lb_killer_num_2->setText(QApplication::translate("Statistics", "Avg killer num:", Q_NULLPTR));
        lb_armor_num_2->setText(QApplication::translate("Statistics", "Avg armor num: ", Q_NULLPTR));
        lb_eye_num_2->setText(QApplication::translate("Statistics", "Avg eye num: ", Q_NULLPTR));
        lb_stationary_organisms->setText(QApplication::translate("Statistics", "Stationary organisms:", Q_NULLPTR));
        lb_organism_size_3->setText(QApplication::translate("Statistics", "Avg organism size:", Q_NULLPTR));
        lb_avg_org_lifetime_3->setText(QApplication::translate("Statistics", "Avg organism lifetime:", Q_NULLPTR));
        lb_avg_age_3->setText(QApplication::translate("Statistics", "Avg organism age:", Q_NULLPTR));
        lb_anatomy_mutation_rate_3->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", Q_NULLPTR));
        lb_brain_mutation_rate_3->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", Q_NULLPTR));
        lb_producer_num_3->setText(QApplication::translate("Statistics", "Avg producer num: ", Q_NULLPTR));
        lb_mouth_num_3->setText(QApplication::translate("Statistics", "Avg mouth num: ", Q_NULLPTR));
        lb_killer_num_3->setText(QApplication::translate("Statistics", "Avg killer num:", Q_NULLPTR));
        lb_armor_num_3->setText(QApplication::translate("Statistics", "Avg armor num: ", Q_NULLPTR));
        lb_eye_num_3->setText(QApplication::translate("Statistics", "Avg eye num: ", Q_NULLPTR));
        lb_organisms_alive_2->setText(QApplication::translate("Statistics", "Organisms alive:", Q_NULLPTR));
        lb_organism_size_4->setText(QApplication::translate("Statistics", "Avg organism size:", Q_NULLPTR));
        lb_avg_org_lifetime_4->setText(QApplication::translate("Statistics", "Avg organism lifetime:", Q_NULLPTR));
        lb_avg_age_4->setText(QApplication::translate("Statistics", "Avg organism age:", Q_NULLPTR));
        lb_anatomy_mutation_rate_4->setText(QApplication::translate("Statistics", "Avg anatomy mutation rate:", Q_NULLPTR));
        lb_brain_mutation_rate_4->setText(QApplication::translate("Statistics", "Avg brain mutation rate:", Q_NULLPTR));
        lb_producer_num_4->setText(QApplication::translate("Statistics", "Avg producer num: ", Q_NULLPTR));
        lb_mover_num_4->setText(QApplication::translate("Statistics", "Avg mover num:", Q_NULLPTR));
        lb_mouth_num_4->setText(QApplication::translate("Statistics", "Avg mouth num: ", Q_NULLPTR));
        lb_killer_num_4->setText(QApplication::translate("Statistics", "Avg killer num:", Q_NULLPTR));
        lb_armor_num_4->setText(QApplication::translate("Statistics", "Avg armor num: ", Q_NULLPTR));
        lb_eye_num_4->setText(QApplication::translate("Statistics", "Avg eye num: ", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class Statistics: public Ui_Statistics {};
} // namespace Ui

QT_END_NAMESPACE

#endif // STATISTICSUI_H
