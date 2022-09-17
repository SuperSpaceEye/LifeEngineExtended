/********************************************************************************
** Form generated from reading UI file 'OCCParameters.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
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
    QCheckBox *checkBox;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *mutation_type_distribution_vert_layout;
    QSpacerItem *verticalSpacer;
    QCheckBox *checkBox_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QLineEdit *lineEdit;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *group_size_weights_layout;
    QSpacerItem *verticalSpacer_2;
    QCheckBox *checkBox_3;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *occ_instructions_weights_layout;
    QSpacerItem *verticalSpacer_3;
    QCheckBox *checkBox_4;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_2;
    QLineEdit *lineEdit_2;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *move_distance_weights_layout;

    void setupUi(QWidget *OCCParametes)
    {
        if (OCCParametes->objectName().isEmpty())
            OCCParametes->setObjectName(QString::fromUtf8("OCCParametes"));
        OCCParametes->resize(729, 546);
        verticalLayout = new QVBoxLayout(OCCParametes);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(OCCParametes);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 727, 544));
        verticalLayout_2 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        checkBox = new QCheckBox(scrollAreaWidgetContents);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));

        verticalLayout_2->addWidget(checkBox);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        mutation_type_distribution_vert_layout = new QVBoxLayout();
        mutation_type_distribution_vert_layout->setObjectName(QString::fromUtf8("mutation_type_distribution_vert_layout"));

        horizontalLayout->addLayout(mutation_type_distribution_vert_layout);


        verticalLayout_2->addLayout(horizontalLayout);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer);

        checkBox_2 = new QCheckBox(scrollAreaWidgetContents);
        checkBox_2->setObjectName(QString::fromUtf8("checkBox_2"));

        verticalLayout_2->addWidget(checkBox_2);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(scrollAreaWidgetContents);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        lineEdit = new QLineEdit(scrollAreaWidgetContents);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        horizontalLayout_2->addWidget(lineEdit);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        group_size_weights_layout = new QVBoxLayout();
        group_size_weights_layout->setObjectName(QString::fromUtf8("group_size_weights_layout"));

        horizontalLayout_3->addLayout(group_size_weights_layout);


        verticalLayout_2->addLayout(horizontalLayout_3);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_2);

        checkBox_3 = new QCheckBox(scrollAreaWidgetContents);
        checkBox_3->setObjectName(QString::fromUtf8("checkBox_3"));

        verticalLayout_2->addWidget(checkBox_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        occ_instructions_weights_layout = new QVBoxLayout();
        occ_instructions_weights_layout->setObjectName(QString::fromUtf8("occ_instructions_weights_layout"));

        horizontalLayout_4->addLayout(occ_instructions_weights_layout);


        verticalLayout_2->addLayout(horizontalLayout_4);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_3);

        checkBox_4 = new QCheckBox(scrollAreaWidgetContents);
        checkBox_4->setObjectName(QString::fromUtf8("checkBox_4"));

        verticalLayout_2->addWidget(checkBox_4);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_2 = new QLabel(scrollAreaWidgetContents);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_5->addWidget(label_2);

        lineEdit_2 = new QLineEdit(scrollAreaWidgetContents);
        lineEdit_2->setObjectName(QString::fromUtf8("lineEdit_2"));

        horizontalLayout_5->addWidget(lineEdit_2);


        verticalLayout_2->addLayout(horizontalLayout_5);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        move_distance_weights_layout = new QVBoxLayout();
        move_distance_weights_layout->setObjectName(QString::fromUtf8("move_distance_weights_layout"));

        horizontalLayout_6->addLayout(move_distance_weights_layout);


        verticalLayout_2->addLayout(horizontalLayout_6);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout->addWidget(scrollArea);


        retranslateUi(OCCParametes);

        QMetaObject::connectSlotsByName(OCCParametes);
    } // setupUi

    void retranslateUi(QWidget *OCCParametes)
    {
        OCCParametes->setWindowTitle(QApplication::translate("OCCParametes", "Organisms Construction Code Parameters", nullptr));
        checkBox->setText(QApplication::translate("OCCParametes", "Use uniform mutation type distribution", nullptr));
        checkBox_2->setText(QApplication::translate("OCCParametes", "Use uniform group size distribution", nullptr));
        label->setText(QApplication::translate("OCCParametes", "Max group size", nullptr));
        checkBox_3->setText(QApplication::translate("OCCParametes", "Use uniform occ instructions distribution", nullptr));
        checkBox_4->setText(QApplication::translate("OCCParametes", "Use uniform move distance distribution", nullptr));
        label_2->setText(QApplication::translate("OCCParametes", "Max move distance", nullptr));
    } // retranslateUi

};

namespace Ui {
    class OCCParametes: public Ui_OCCParametes {};
} // namespace Ui

QT_END_NAMESPACE

#endif // OCCPARAMETERSUI_H
