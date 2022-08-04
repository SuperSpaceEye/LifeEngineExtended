/********************************************************************************
** Form generated from reading UI file 'ChangeValueEventNodeWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef CHANGEVALUEEVENTNODEWIDGETUI_H
#define CHANGEVALUEEVENTNODEWIDGETUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ChangeValueEventNodeWidget
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_3;
    QComboBox *cmb_change_value;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QLineEdit *le_target_value;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_4;
    QComboBox *cmb_change_mode;
    QHBoxLayout *time_horizon_layout;
    QLabel *label_5;
    QLineEdit *le_time_horizon;

    void setupUi(QWidget *ChangeValueEventNodeWidget)
    {
        if (ChangeValueEventNodeWidget->objectName().isEmpty())
            ChangeValueEventNodeWidget->setObjectName(QString::fromUtf8("ChangeValueEventNodeWidget"));
        ChangeValueEventNodeWidget->resize(512, 301);
        verticalLayout = new QVBoxLayout(ChangeValueEventNodeWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(ChangeValueEventNodeWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(ChangeValueEventNodeWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        cmb_change_value = new QComboBox(ChangeValueEventNodeWidget);
        cmb_change_value->setObjectName(QString::fromUtf8("cmb_change_value"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(cmb_change_value->sizePolicy().hasHeightForWidth());
        cmb_change_value->setSizePolicy(sizePolicy);

        horizontalLayout_2->addWidget(cmb_change_value);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(ChangeValueEventNodeWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        le_target_value = new QLineEdit(ChangeValueEventNodeWidget);
        le_target_value->setObjectName(QString::fromUtf8("le_target_value"));

        horizontalLayout->addWidget(le_target_value);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_4 = new QLabel(ChangeValueEventNodeWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_3->addWidget(label_4);

        cmb_change_mode = new QComboBox(ChangeValueEventNodeWidget);
        cmb_change_mode->addItem(QString());
        cmb_change_mode->addItem(QString());
        cmb_change_mode->setObjectName(QString::fromUtf8("cmb_change_mode"));
        sizePolicy.setHeightForWidth(cmb_change_mode->sizePolicy().hasHeightForWidth());
        cmb_change_mode->setSizePolicy(sizePolicy);

        horizontalLayout_3->addWidget(cmb_change_mode);


        verticalLayout->addLayout(horizontalLayout_3);

        time_horizon_layout = new QHBoxLayout();
        time_horizon_layout->setObjectName(QString::fromUtf8("time_horizon_layout"));
        label_5 = new QLabel(ChangeValueEventNodeWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        time_horizon_layout->addWidget(label_5);

        le_time_horizon = new QLineEdit(ChangeValueEventNodeWidget);
        le_time_horizon->setObjectName(QString::fromUtf8("le_time_horizon"));

        time_horizon_layout->addWidget(le_time_horizon);


        verticalLayout->addLayout(time_horizon_layout);


        retranslateUi(ChangeValueEventNodeWidget);
        QObject::connect(cmb_change_mode, SIGNAL(currentTextChanged(QString)), ChangeValueEventNodeWidget, SLOT(cmb_change_mode_slot(QString)));
        QObject::connect(cmb_change_value, SIGNAL(currentTextChanged(QString)), ChangeValueEventNodeWidget, SLOT(cmb_change_value_slot(QString)));
        QObject::connect(le_target_value, SIGNAL(returnPressed()), ChangeValueEventNodeWidget, SLOT(le_target_value_slot()));
        QObject::connect(le_time_horizon, SIGNAL(returnPressed()), ChangeValueEventNodeWidget, SLOT(le_time_horizon_slot()));

        QMetaObject::connectSlotsByName(ChangeValueEventNodeWidget);
    } // setupUi

    void retranslateUi(QWidget *ChangeValueEventNodeWidget)
    {
        ChangeValueEventNodeWidget->setWindowTitle(QApplication::translate("ChangeValueEventNodeWidget", "Form", nullptr));
        label->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change Value", nullptr));
        label_3->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change value", nullptr));
        label_2->setText(QApplication::translate("ChangeValueEventNodeWidget", "Target value", nullptr));
        label_4->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change mode", nullptr));
        cmb_change_mode->setItemText(0, QApplication::translate("ChangeValueEventNodeWidget", "Linear", nullptr));
        cmb_change_mode->setItemText(1, QApplication::translate("ChangeValueEventNodeWidget", "Step", nullptr));

        label_5->setText(QApplication::translate("ChangeValueEventNodeWidget", "Time horizon", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ChangeValueEventNodeWidget: public Ui_ChangeValueEventNodeWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // CHANGEVALUEEVENTNODEWIDGETUI_H
