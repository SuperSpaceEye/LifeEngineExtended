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
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ChangeValueEventNodeWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_5;
    QPushButton *b_new_event_left;
    QSpacerItem *horizontalSpacer;
    QPushButton *b_delete_event;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *b_new_event_right;
    QLabel *label;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_6;
    QLineEdit *le_update_every_n_ticks;
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
    QLabel *time_horizon_label;
    QLineEdit *le_time_horizon;

    void setupUi(QWidget *ChangeValueEventNodeWidget)
    {
        if (ChangeValueEventNodeWidget->objectName().isEmpty())
            ChangeValueEventNodeWidget->setObjectName(QString::fromUtf8("ChangeValueEventNodeWidget"));
        ChangeValueEventNodeWidget->resize(400, 200);
        verticalLayout = new QVBoxLayout(ChangeValueEventNodeWidget);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(ChangeValueEventNodeWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::WinPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frame);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        b_new_event_left = new QPushButton(frame);
        b_new_event_left->setObjectName(QString::fromUtf8("b_new_event_left"));

        horizontalLayout_5->addWidget(b_new_event_left);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer);

        b_delete_event = new QPushButton(frame);
        b_delete_event->setObjectName(QString::fromUtf8("b_delete_event"));

        horizontalLayout_5->addWidget(b_delete_event);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_2);

        b_new_event_right = new QPushButton(frame);
        b_new_event_right->setObjectName(QString::fromUtf8("b_new_event_right"));

        horizontalLayout_5->addWidget(b_new_event_right);


        verticalLayout_2->addLayout(horizontalLayout_5);

        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(label);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_6 = new QLabel(frame);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_4->addWidget(label_6);

        le_update_every_n_ticks = new QLineEdit(frame);
        le_update_every_n_ticks->setObjectName(QString::fromUtf8("le_update_every_n_ticks"));

        horizontalLayout_4->addWidget(le_update_every_n_ticks);


        verticalLayout_2->addLayout(horizontalLayout_4);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(frame);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        cmb_change_value = new QComboBox(frame);
        cmb_change_value->setObjectName(QString::fromUtf8("cmb_change_value"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(cmb_change_value->sizePolicy().hasHeightForWidth());
        cmb_change_value->setSizePolicy(sizePolicy);

        horizontalLayout_2->addWidget(cmb_change_value);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(frame);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        le_target_value = new QLineEdit(frame);
        le_target_value->setObjectName(QString::fromUtf8("le_target_value"));

        horizontalLayout->addWidget(le_target_value);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_4 = new QLabel(frame);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_3->addWidget(label_4);

        cmb_change_mode = new QComboBox(frame);
        cmb_change_mode->addItem(QString());
        cmb_change_mode->addItem(QString());
        cmb_change_mode->setObjectName(QString::fromUtf8("cmb_change_mode"));
        sizePolicy.setHeightForWidth(cmb_change_mode->sizePolicy().hasHeightForWidth());
        cmb_change_mode->setSizePolicy(sizePolicy);

        horizontalLayout_3->addWidget(cmb_change_mode);


        verticalLayout_2->addLayout(horizontalLayout_3);

        time_horizon_layout = new QHBoxLayout();
        time_horizon_layout->setObjectName(QString::fromUtf8("time_horizon_layout"));
        time_horizon_label = new QLabel(frame);
        time_horizon_label->setObjectName(QString::fromUtf8("time_horizon_label"));

        time_horizon_layout->addWidget(time_horizon_label);

        le_time_horizon = new QLineEdit(frame);
        le_time_horizon->setObjectName(QString::fromUtf8("le_time_horizon"));

        time_horizon_layout->addWidget(le_time_horizon);


        verticalLayout_2->addLayout(time_horizon_layout);


        verticalLayout->addWidget(frame);


        retranslateUi(ChangeValueEventNodeWidget);
        QObject::connect(cmb_change_mode, SIGNAL(currentTextChanged(QString)), ChangeValueEventNodeWidget, SLOT(cmb_change_mode_slot(QString)));
        QObject::connect(cmb_change_value, SIGNAL(currentTextChanged(QString)), ChangeValueEventNodeWidget, SLOT(cmb_change_value_slot(QString)));
        QObject::connect(le_target_value, SIGNAL(returnPressed()), ChangeValueEventNodeWidget, SLOT(le_target_value_slot()));
        QObject::connect(le_time_horizon, SIGNAL(returnPressed()), ChangeValueEventNodeWidget, SLOT(le_time_horizon_slot()));
        QObject::connect(le_update_every_n_ticks, SIGNAL(returnPressed()), ChangeValueEventNodeWidget, SLOT(le_update_every_n_ticks_slot()));
        QObject::connect(b_new_event_left, SIGNAL(clicked()), ChangeValueEventNodeWidget, SLOT(b_new_event_left_slot()));
        QObject::connect(b_new_event_right, SIGNAL(clicked()), ChangeValueEventNodeWidget, SLOT(b_new_event_right_slot()));
        QObject::connect(b_delete_event, SIGNAL(clicked()), ChangeValueEventNodeWidget, SLOT(b_delete_event_slot()));

        QMetaObject::connectSlotsByName(ChangeValueEventNodeWidget);
    } // setupUi

    void retranslateUi(QWidget *ChangeValueEventNodeWidget)
    {
        ChangeValueEventNodeWidget->setWindowTitle(QApplication::translate("ChangeValueEventNodeWidget", "Form", nullptr));
        b_new_event_left->setText(QApplication::translate("ChangeValueEventNodeWidget", "<= New event", nullptr));
        b_delete_event->setText(QApplication::translate("ChangeValueEventNodeWidget", "Delete Event", nullptr));
        b_new_event_right->setText(QApplication::translate("ChangeValueEventNodeWidget", "New event =>", nullptr));
        label->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change Value", nullptr));
        label_6->setText(QApplication::translate("ChangeValueEventNodeWidget", "Update every n ticks", nullptr));
        label_3->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change value", nullptr));
        label_2->setText(QApplication::translate("ChangeValueEventNodeWidget", "Target value", nullptr));
        label_4->setText(QApplication::translate("ChangeValueEventNodeWidget", "Change mode", nullptr));
        cmb_change_mode->setItemText(0, QApplication::translate("ChangeValueEventNodeWidget", "Linear", nullptr));
        cmb_change_mode->setItemText(1, QApplication::translate("ChangeValueEventNodeWidget", "Step", nullptr));

        time_horizon_label->setText(QApplication::translate("ChangeValueEventNodeWidget", "Time horizon", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ChangeValueEventNodeWidget: public Ui_ChangeValueEventNodeWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // CHANGEVALUEEVENTNODEWIDGETUI_H
