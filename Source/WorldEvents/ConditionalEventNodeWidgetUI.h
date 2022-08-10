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
            ConditionalEventNode->setObjectName(QString::fromUtf8("ConditionalEventNode"));
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
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(ConditionalEventNode);
        frame->setObjectName(QString::fromUtf8("frame"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
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
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(4);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        b_new_event_left = new QPushButton(frame);
        b_new_event_left->setObjectName(QString::fromUtf8("b_new_event_left"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(b_new_event_left->sizePolicy().hasHeightForWidth());
        b_new_event_left->setSizePolicy(sizePolicy2);

        horizontalLayout_4->addWidget(b_new_event_left);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        b_delete_event = new QPushButton(frame);
        b_delete_event->setObjectName(QString::fromUtf8("b_delete_event"));

        horizontalLayout_4->addWidget(b_delete_event);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_2);

        b_new_event_right = new QPushButton(frame);
        b_new_event_right->setObjectName(QString::fromUtf8("b_new_event_right"));
        sizePolicy2.setHeightForWidth(b_new_event_right->sizePolicy().hasHeightForWidth());
        b_new_event_right->setSizePolicy(sizePolicy2);

        horizontalLayout_4->addWidget(b_new_event_right);


        verticalLayout_2->addLayout(horizontalLayout_4);

        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));
        sizePolicy2.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy2);
        label->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(label);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_6 = new QLabel(frame);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        sizePolicy2.setHeightForWidth(label_6->sizePolicy().hasHeightForWidth());
        label_6->setSizePolicy(sizePolicy2);

        horizontalLayout_3->addWidget(label_6);

        le_update_every_n_ticks = new QLineEdit(frame);
        le_update_every_n_ticks->setObjectName(QString::fromUtf8("le_update_every_n_ticks"));

        horizontalLayout_3->addWidget(le_update_every_n_ticks);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(frame);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        sizePolicy2.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy2);

        horizontalLayout_2->addWidget(label_3);

        cmb_condition_mode = new QComboBox(frame);
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->addItem(QString());
        cmb_condition_mode->setObjectName(QString::fromUtf8("cmb_condition_mode"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(cmb_condition_mode->sizePolicy().hasHeightForWidth());
        cmb_condition_mode->setSizePolicy(sizePolicy3);

        horizontalLayout_2->addWidget(cmb_condition_mode);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(frame);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        cmb_condition_value = new QComboBox(frame);
        cmb_condition_value->setObjectName(QString::fromUtf8("cmb_condition_value"));
        sizePolicy3.setHeightForWidth(cmb_condition_value->sizePolicy().hasHeightForWidth());
        cmb_condition_value->setSizePolicy(sizePolicy3);

        horizontalLayout->addWidget(cmb_condition_value);

        label_condition = new QLabel(frame);
        label_condition->setObjectName(QString::fromUtf8("label_condition"));
        sizePolicy2.setHeightForWidth(label_condition->sizePolicy().hasHeightForWidth());
        label_condition->setSizePolicy(sizePolicy2);

        horizontalLayout->addWidget(label_condition);

        le_value_to_compare_against = new QLineEdit(frame);
        le_value_to_compare_against->setObjectName(QString::fromUtf8("le_value_to_compare_against"));

        horizontalLayout->addWidget(le_value_to_compare_against);

        label_4 = new QLabel(frame);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout->addWidget(label_4);


        verticalLayout_2->addLayout(horizontalLayout);

        label_5 = new QLabel(frame);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setEnabled(false);
        sizePolicy2.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy2);
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
        ConditionalEventNode->setWindowTitle(QApplication::translate("ConditionalEventNode", "Form", nullptr));
        b_new_event_left->setText(QApplication::translate("ConditionalEventNode", "<=New event", nullptr));
        b_delete_event->setText(QApplication::translate("ConditionalEventNode", "Delete event", nullptr));
        b_new_event_right->setText(QApplication::translate("ConditionalEventNode", "New Event=>", nullptr));
        label->setText(QApplication::translate("ConditionalEventNode", "Condition", nullptr));
        label_6->setText(QApplication::translate("ConditionalEventNode", "Update every n ticks ", nullptr));
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
