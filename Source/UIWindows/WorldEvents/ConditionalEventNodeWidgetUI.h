/********************************************************************************
** Form generated from reading UI file 'ConditionalEventNodeWidget.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef CONDITIONALEVENTNODEWIDGETUI_H
#define CONDITIONALEVENTNODEWIDGETUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ConditionalEventNode
{
public:
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *b_new_event_left;
    QSpacerItem *horizontalSpacer;
    QPushButton *b_delete_event;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *b_new_event_right;
    QLabel *label;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_6;
    QLineEdit *le_update_every_n_ticks;
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
            ConditionalEventNode->setObjectName("ConditionalEventNode");
        ConditionalEventNode->setWindowModality(Qt::NonModal);
        ConditionalEventNode->resize(400, 200);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(ConditionalEventNode->sizePolicy().hasHeightForWidth());
        ConditionalEventNode->setSizePolicy(sizePolicy);
        ConditionalEventNode->setFocusPolicy(Qt::NoFocus);
        ConditionalEventNode->setContextMenuPolicy(Qt::DefaultContextMenu);
        verticalLayout = new QVBoxLayout(ConditionalEventNode);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(ConditionalEventNode);
        frame->setObjectName("frame");
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy1);
        frame->setFrameShape(QFrame::WinPanel);
        frame->setFrameShadow(QFrame::Raised);
        frame->setLineWidth(0);
        frame->setMidLineWidth(0);
        verticalLayout_2 = new QVBoxLayout(frame);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(4);
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        b_new_event_left = new QPushButton(frame);
        b_new_event_left->setObjectName("b_new_event_left");
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(b_new_event_left->sizePolicy().hasHeightForWidth());
        b_new_event_left->setSizePolicy(sizePolicy2);

        horizontalLayout_4->addWidget(b_new_event_left);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        b_delete_event = new QPushButton(frame);
        b_delete_event->setObjectName("b_delete_event");

        horizontalLayout_4->addWidget(b_delete_event);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_2);

        b_new_event_right = new QPushButton(frame);
        b_new_event_right->setObjectName("b_new_event_right");
        sizePolicy2.setHeightForWidth(b_new_event_right->sizePolicy().hasHeightForWidth());
        b_new_event_right->setSizePolicy(sizePolicy2);

        horizontalLayout_4->addWidget(b_new_event_right);


        verticalLayout_2->addLayout(horizontalLayout_4);

        label = new QLabel(frame);
        label->setObjectName("label");
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy3);
        label->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(label);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        label_6 = new QLabel(frame);
        label_6->setObjectName("label_6");
        sizePolicy3.setHeightForWidth(label_6->sizePolicy().hasHeightForWidth());
        label_6->setSizePolicy(sizePolicy3);

        horizontalLayout_3->addWidget(label_6);

        le_update_every_n_ticks = new QLineEdit(frame);
        le_update_every_n_ticks->setObjectName("le_update_every_n_ticks");
        QSizePolicy sizePolicy4(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(le_update_every_n_ticks->sizePolicy().hasHeightForWidth());
        le_update_every_n_ticks->setSizePolicy(sizePolicy4);

        horizontalLayout_3->addWidget(le_update_every_n_ticks);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        label_3 = new QLabel(frame);
        label_3->setObjectName("label_3");
        sizePolicy3.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy3);

        horizontalLayout_2->addWidget(label_3);

        cmb_condition_mode = new QComboBox(frame);
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->setObjectName("cmb_condition_mode");
        sizePolicy4.setHeightForWidth(cmb_condition_mode->sizePolicy().hasHeightForWidth());
        cmb_condition_mode->setSizePolicy(sizePolicy4);

        horizontalLayout_2->addWidget(cmb_condition_mode);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        label_2 = new QLabel(frame);
        label_2->setObjectName("label_2");

        horizontalLayout->addWidget(label_2);

        cmb_condition_value = new QComboBox(frame);
        cmb_condition_value->setObjectName("cmb_condition_value");
        sizePolicy4.setHeightForWidth(cmb_condition_value->sizePolicy().hasHeightForWidth());
        cmb_condition_value->setSizePolicy(sizePolicy4);

        horizontalLayout->addWidget(cmb_condition_value);

        label_condition = new QLabel(frame);
        label_condition->setObjectName("label_condition");
        sizePolicy3.setHeightForWidth(label_condition->sizePolicy().hasHeightForWidth());
        label_condition->setSizePolicy(sizePolicy3);

        horizontalLayout->addWidget(label_condition);

        le_value_to_compare_against = new QLineEdit(frame);
        le_value_to_compare_against->setObjectName("le_value_to_compare_against");
        sizePolicy4.setHeightForWidth(le_value_to_compare_against->sizePolicy().hasHeightForWidth());
        le_value_to_compare_against->setSizePolicy(sizePolicy4);

        horizontalLayout->addWidget(le_value_to_compare_against);

        label_4 = new QLabel(frame);
        label_4->setObjectName("label_4");

        horizontalLayout->addWidget(label_4);


        verticalLayout_2->addLayout(horizontalLayout);

        label_5 = new QLabel(frame);
        label_5->setObjectName("label_5");
        label_5->setEnabled(false);
        sizePolicy3.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy3);
        label_5->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(label_5);


        verticalLayout->addWidget(frame);


        retranslateUi(ConditionalEventNode);
        QObject::connect(cmb_condition_mode, SIGNAL(currentTextChanged(QString)), ConditionalEventNode, SLOT(cmb_condition_mode_slot(QString)));
        QObject::connect(cmb_condition_value, SIGNAL(currentTextChanged(QString)), ConditionalEventNode, SLOT(cmb_condition_value_slot(QString)));
        QObject::connect(le_value_to_compare_against, SIGNAL(returnPressed()), ConditionalEventNode, SLOT(le_value_to_compare_against_slot()));
        QObject::connect(le_update_every_n_ticks, SIGNAL(returnPressed()), ConditionalEventNode, SLOT(le_update_every_n_ticks_slot()));
        QObject::connect(b_new_event_left, SIGNAL(clicked()), ConditionalEventNode, SLOT(b_new_event_left_slot()));
        QObject::connect(b_new_event_right, SIGNAL(clicked()), ConditionalEventNode, SLOT(b_new_event_right_slot()));
        QObject::connect(b_delete_event, SIGNAL(clicked()), ConditionalEventNode, SLOT(b_delete_event_slot()));

        QMetaObject::connectSlotsByName(ConditionalEventNode);
    } // setupUi

    void retranslateUi(QWidget *ConditionalEventNode)
    {
        ConditionalEventNode->setWindowTitle(QCoreApplication::translate("ConditionalEventNode", "Form", nullptr));
        b_new_event_left->setText(QCoreApplication::translate("ConditionalEventNode", "<=New event", nullptr));
        b_delete_event->setText(QCoreApplication::translate("ConditionalEventNode", "Delete event", nullptr));
        b_new_event_right->setText(QCoreApplication::translate("ConditionalEventNode", "New Event=>", nullptr));
        label->setText(QCoreApplication::translate("ConditionalEventNode", "Condition", nullptr));
        label_6->setText(QCoreApplication::translate("ConditionalEventNode", "Update every n ticks ", nullptr));
        label_3->setText(QCoreApplication::translate("ConditionalEventNode", "Condition mode", nullptr));
        cmb_condition_mode->setItemText(0, QCoreApplication::translate("ConditionalEventNode", "More or Equal", nullptr));
        cmb_condition_mode->setItemText(1, QCoreApplication::translate("ConditionalEventNode", "Less or Equal", nullptr));

        label_2->setText(QCoreApplication::translate("ConditionalEventNode", "If ", nullptr));
        label_condition->setText(QCoreApplication::translate("ConditionalEventNode", ">=", nullptr));
        label_4->setText(QCoreApplication::translate("ConditionalEventNode", "Then =>", nullptr));
        label_5->setText(QCoreApplication::translate("ConditionalEventNode", "Else\n"
"||\n"
"V", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ConditionalEventNode: public Ui_ConditionalEventNode {};
} // namespace Ui

QT_END_NAMESPACE

#endif // CONDITIONALEVENTNODEWIDGETUI_H
