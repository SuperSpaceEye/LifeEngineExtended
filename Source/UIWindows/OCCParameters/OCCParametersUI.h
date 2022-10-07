/********************************************************************************
** Form generated from reading UI file 'OCCParameters.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef OCCPARAMETERSUI_H
#define OCCPARAMETERSUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OCCParametes
{
public:
    QVBoxLayout *verticalLayout;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *cb_use_uniform_mutation_type;
    QScrollArea *scrollArea_4;
    QWidget *scrollAreaWidgetContents_4;
    QVBoxLayout *verticalLayout_5;
    QWidget *mutation_types_widget;
    QVBoxLayout *verticalLayout_7;
    QSpacerItem *verticalSpacer;
    QCheckBox *cb_use_uniform_group_size;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QLineEdit *le_max_group_size;
    QScrollArea *scrollArea_3;
    QWidget *scrollAreaWidgetContents_3;
    QVBoxLayout *verticalLayout_4;
    QWidget *group_size_widget;
    QVBoxLayout *verticalLayout_6;
    QVBoxLayout *group_size_layout;
    QSpacerItem *verticalSpacer_2;
    QCheckBox *cb_use_uniform_occ_instructions;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QVBoxLayout *verticalLayout_3;
    QWidget *occ_mutation_type_widget;
    QVBoxLayout *occ_instructions_weights_layout;
    QSpacerItem *verticalSpacer_3;
    QCheckBox *cb_use_uniform_move_distance;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_2;
    QLineEdit *le_max_move_distance;
    QScrollArea *scrollArea_5;
    QWidget *scrollAreaWidgetContents_5;
    QVBoxLayout *verticalLayout_8;
    QWidget *move_distance_widget;
    QVBoxLayout *verticalLayout_9;
    QVBoxLayout *move_distance_layout;

    void setupUi(QWidget *OCCParametes)
    {
        if (OCCParametes->objectName().isEmpty())
            OCCParametes->setObjectName("OCCParametes");
        OCCParametes->resize(808, 746);
        verticalLayout = new QVBoxLayout(OCCParametes);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(OCCParametes);
        scrollArea->setObjectName("scrollArea");
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName("scrollAreaWidgetContents");
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 806, 744));
        verticalLayout_2 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_2->setObjectName("verticalLayout_2");
        cb_use_uniform_mutation_type = new QCheckBox(scrollAreaWidgetContents);
        cb_use_uniform_mutation_type->setObjectName("cb_use_uniform_mutation_type");

        verticalLayout_2->addWidget(cb_use_uniform_mutation_type);

        scrollArea_4 = new QScrollArea(scrollAreaWidgetContents);
        scrollArea_4->setObjectName("scrollArea_4");
        scrollArea_4->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName("scrollAreaWidgetContents_4");
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 786, 95));
        verticalLayout_5 = new QVBoxLayout(scrollAreaWidgetContents_4);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setObjectName("verticalLayout_5");
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        mutation_types_widget = new QWidget(scrollAreaWidgetContents_4);
        mutation_types_widget->setObjectName("mutation_types_widget");
        verticalLayout_7 = new QVBoxLayout(mutation_types_widget);
        verticalLayout_7->setObjectName("verticalLayout_7");

        verticalLayout_5->addWidget(mutation_types_widget);

        scrollArea_4->setWidget(scrollAreaWidgetContents_4);

        verticalLayout_2->addWidget(scrollArea_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer);

        cb_use_uniform_group_size = new QCheckBox(scrollAreaWidgetContents);
        cb_use_uniform_group_size->setObjectName("cb_use_uniform_group_size");

        verticalLayout_2->addWidget(cb_use_uniform_group_size);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        label = new QLabel(scrollAreaWidgetContents);
        label->setObjectName("label");

        horizontalLayout_2->addWidget(label);

        le_max_group_size = new QLineEdit(scrollAreaWidgetContents);
        le_max_group_size->setObjectName("le_max_group_size");

        horizontalLayout_2->addWidget(le_max_group_size);


        verticalLayout_2->addLayout(horizontalLayout_2);

        scrollArea_3 = new QScrollArea(scrollAreaWidgetContents);
        scrollArea_3->setObjectName("scrollArea_3");
        scrollArea_3->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName("scrollAreaWidgetContents_3");
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 786, 95));
        verticalLayout_4 = new QVBoxLayout(scrollAreaWidgetContents_3);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName("verticalLayout_4");
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        group_size_widget = new QWidget(scrollAreaWidgetContents_3);
        group_size_widget->setObjectName("group_size_widget");
        verticalLayout_6 = new QVBoxLayout(group_size_widget);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName("verticalLayout_6");
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        group_size_layout = new QVBoxLayout();
        group_size_layout->setObjectName("group_size_layout");
        group_size_layout->setContentsMargins(9, 9, 9, 9);

        verticalLayout_6->addLayout(group_size_layout);


        verticalLayout_4->addWidget(group_size_widget);

        scrollArea_3->setWidget(scrollAreaWidgetContents_3);

        verticalLayout_2->addWidget(scrollArea_3);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_2);

        cb_use_uniform_occ_instructions = new QCheckBox(scrollAreaWidgetContents);
        cb_use_uniform_occ_instructions->setObjectName("cb_use_uniform_occ_instructions");

        verticalLayout_2->addWidget(cb_use_uniform_occ_instructions);

        scrollArea_2 = new QScrollArea(scrollAreaWidgetContents);
        scrollArea_2->setObjectName("scrollArea_2");
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(scrollArea_2->sizePolicy().hasHeightForWidth());
        scrollArea_2->setSizePolicy(sizePolicy);
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName("scrollAreaWidgetContents_2");
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 786, 95));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(scrollAreaWidgetContents_2->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_2->setSizePolicy(sizePolicy1);
        verticalLayout_3 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName("verticalLayout_3");
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        occ_mutation_type_widget = new QWidget(scrollAreaWidgetContents_2);
        occ_mutation_type_widget->setObjectName("occ_mutation_type_widget");
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(40);
        sizePolicy2.setVerticalStretch(40);
        sizePolicy2.setHeightForWidth(occ_mutation_type_widget->sizePolicy().hasHeightForWidth());
        occ_mutation_type_widget->setSizePolicy(sizePolicy2);
        occ_instructions_weights_layout = new QVBoxLayout(occ_mutation_type_widget);
        occ_instructions_weights_layout->setObjectName("occ_instructions_weights_layout");

        verticalLayout_3->addWidget(occ_mutation_type_widget);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        verticalLayout_2->addWidget(scrollArea_2);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_3);

        cb_use_uniform_move_distance = new QCheckBox(scrollAreaWidgetContents);
        cb_use_uniform_move_distance->setObjectName("cb_use_uniform_move_distance");

        verticalLayout_2->addWidget(cb_use_uniform_move_distance);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        label_2 = new QLabel(scrollAreaWidgetContents);
        label_2->setObjectName("label_2");

        horizontalLayout_5->addWidget(label_2);

        le_max_move_distance = new QLineEdit(scrollAreaWidgetContents);
        le_max_move_distance->setObjectName("le_max_move_distance");

        horizontalLayout_5->addWidget(le_max_move_distance);


        verticalLayout_2->addLayout(horizontalLayout_5);

        scrollArea_5 = new QScrollArea(scrollAreaWidgetContents);
        scrollArea_5->setObjectName("scrollArea_5");
        scrollArea_5->setWidgetResizable(true);
        scrollAreaWidgetContents_5 = new QWidget();
        scrollAreaWidgetContents_5->setObjectName("scrollAreaWidgetContents_5");
        scrollAreaWidgetContents_5->setGeometry(QRect(0, 0, 786, 95));
        verticalLayout_8 = new QVBoxLayout(scrollAreaWidgetContents_5);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName("verticalLayout_8");
        verticalLayout_8->setContentsMargins(0, 0, 0, 0);
        move_distance_widget = new QWidget(scrollAreaWidgetContents_5);
        move_distance_widget->setObjectName("move_distance_widget");
        verticalLayout_9 = new QVBoxLayout(move_distance_widget);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setObjectName("verticalLayout_9");
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        move_distance_layout = new QVBoxLayout();
        move_distance_layout->setObjectName("move_distance_layout");
        move_distance_layout->setContentsMargins(9, 9, 9, 9);

        verticalLayout_9->addLayout(move_distance_layout);


        verticalLayout_8->addWidget(move_distance_widget);

        scrollArea_5->setWidget(scrollAreaWidgetContents_5);

        verticalLayout_2->addWidget(scrollArea_5);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout->addWidget(scrollArea);


        retranslateUi(OCCParametes);
        QObject::connect(cb_use_uniform_group_size, SIGNAL(toggled(bool)), OCCParametes, SLOT(cb_use_uniform_group_size_slot(bool)));
        QObject::connect(cb_use_uniform_move_distance, SIGNAL(toggled(bool)), OCCParametes, SLOT(cb_use_uniform_move_distance_slot(bool)));
        QObject::connect(cb_use_uniform_mutation_type, SIGNAL(toggled(bool)), OCCParametes, SLOT(cb_use_uniform_mutation_type_slot(bool)));
        QObject::connect(cb_use_uniform_occ_instructions, SIGNAL(toggled(bool)), OCCParametes, SLOT(cb_use_uniform_occ_instructions_slot(bool)));
        QObject::connect(le_max_group_size, SIGNAL(returnPressed()), OCCParametes, SLOT(le_max_group_size_slot()));
        QObject::connect(le_max_move_distance, SIGNAL(returnPressed()), OCCParametes, SLOT(le_max_move_distance_slot()));

        QMetaObject::connectSlotsByName(OCCParametes);
    } // setupUi

    void retranslateUi(QWidget *OCCParametes)
    {
        OCCParametes->setWindowTitle(QCoreApplication::translate("OCCParametes", "Organisms Construction Code Parameters", nullptr));
        cb_use_uniform_mutation_type->setText(QCoreApplication::translate("OCCParametes", "Use uniform mutation type distribution", nullptr));
        cb_use_uniform_group_size->setText(QCoreApplication::translate("OCCParametes", "Use uniform group size distribution", nullptr));
        label->setText(QCoreApplication::translate("OCCParametes", "Max group size", nullptr));
        cb_use_uniform_occ_instructions->setText(QCoreApplication::translate("OCCParametes", "Use uniform occ instructions distribution", nullptr));
        cb_use_uniform_move_distance->setText(QCoreApplication::translate("OCCParametes", "Use uniform move distance distribution", nullptr));
        label_2->setText(QCoreApplication::translate("OCCParametes", "Max move distance", nullptr));
    } // retranslateUi

};

namespace Ui {
    class OCCParametes: public Ui_OCCParametes {};
} // namespace Ui

QT_END_NAMESPACE

#endif // OCCPARAMETERSUI_H
