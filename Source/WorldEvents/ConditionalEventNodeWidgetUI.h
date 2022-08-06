/********************************************************************************
** Form generated from reading UI file 'ConditionalEventNodeWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef CONDITIONALEVENTNODEWIDGETUI_H
#define CONDITIONALEVENTNODEWIDGETUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ConditionalEventNode
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_3;
    QComboBox *cmb_condition_mode;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QComboBox *cmb_condition_value;
    QLabel *label_condition;
    QLineEdit *le_value_to_compare_against;
    QLabel *label_4;
    QLabel *label_5;

    void setupUi(QWidget *ConditionalEventNode)
    {
        if (ConditionalEventNode->objectName().isEmpty())
            ConditionalEventNode->setObjectName(QString::fromUtf8("ConditionalEventNode"));
        ConditionalEventNode->resize(400, 200);
        verticalLayout = new QVBoxLayout(ConditionalEventNode);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(ConditionalEventNode);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(ConditionalEventNode);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        cmb_condition_mode = new QComboBox(ConditionalEventNode);
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->setObjectName(QString::fromUtf8("cmb_condition_mode"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(cmb_condition_mode->sizePolicy().hasHeightForWidth());
        cmb_condition_mode->setSizePolicy(sizePolicy);

        horizontalLayout_2->addWidget(cmb_condition_mode);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(ConditionalEventNode);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        cmb_condition_value = new QComboBox(ConditionalEventNode);
        cmb_condition_value->setObjectName(QString::fromUtf8("cmb_condition_value"));
        sizePolicy.setHeightForWidth(cmb_condition_value->sizePolicy().hasHeightForWidth());
        cmb_condition_value->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(cmb_condition_value);

        label_condition = new QLabel(ConditionalEventNode);
        label_condition->setObjectName(QString::fromUtf8("label_condition"));

        horizontalLayout->addWidget(label_condition);

        le_value_to_compare_against = new QLineEdit(ConditionalEventNode);
        le_value_to_compare_against->setObjectName(QString::fromUtf8("le_value_to_compare_against"));

        horizontalLayout->addWidget(le_value_to_compare_against);

        label_4 = new QLabel(ConditionalEventNode);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout->addWidget(label_4);


        verticalLayout->addLayout(horizontalLayout);

        label_5 = new QLabel(ConditionalEventNode);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label_5);


        retranslateUi(ConditionalEventNode);
        QObject::connect(cmb_condition_mode, SIGNAL(currentTextChanged(QString)), ConditionalEventNode, SLOT(cmb_condition_mode_slot(QString)));
        QObject::connect(cmb_condition_value, SIGNAL(currentTextChanged(QString)), ConditionalEventNode, SLOT(cmb_condition_value_slot(QString)));
        QObject::connect(le_value_to_compare_against, SIGNAL(returnPressed()), ConditionalEventNode, SLOT(le_value_to_compare_against_slot()));

        QMetaObject::connectSlotsByName(ConditionalEventNode);
    } // setupUi

    void retranslateUi(QWidget *ConditionalEventNode)
    {
        ConditionalEventNode->setWindowTitle(QApplication::translate("ConditionalEventNode", "Form", nullptr));
        label->setText(QApplication::translate("ConditionalEventNode", "Condition", nullptr));
        label_3->setText(QApplication::translate("ConditionalEventNode", "Condition mode", nullptr));
        cmb_condition_mode->setItemText(0, QApplication::translate("ConditionalEventNode", "More or Equal", nullptr));
        cmb_condition_mode->setItemText(1, QApplication::translate("ConditionalEventNode", "Less or Equal", nullptr));

        label_2->setText(QApplication::translate("ConditionalEventNode", "If ", nullptr));
        label_condition->setText(QApplication::translate("ConditionalEventNode", ">=", nullptr));
        label_4->setText(QApplication::translate("ConditionalEventNode", "Then =>", nullptr));
        label_5->setText(QApplication::translate("ConditionalEventNode", "Else\n"
"||\n"
"V", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ConditionalEventNode: public Ui_ConditionalEventNode {};
} // namespace Ui

QT_END_NAMESPACE

#endif // CONDITIONALEVENTNODEWIDGETUI_H
