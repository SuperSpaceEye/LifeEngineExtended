/********************************************************************************
** Form generated from reading UI file 'OCCInstructionWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef OCCINSTRUCTIONWIDGETUI_H
#define OCCINSTRUCTIONWIDGETUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OCCInstWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QPushButton *b_insert_left;
    QPushButton *b_delete_inst;
    QPushButton *b_insert_right;
    QSpacerItem *verticalSpacer;
    QComboBox *cmb_occ_instructions;
    QLabel *name_label;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *OCCInstWidget)
    {
        if (OCCInstWidget->objectName().isEmpty())
            OCCInstWidget->setObjectName(QString::fromUtf8("OCCInstWidget"));
        OCCInstWidget->resize(280, 195);
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(OCCInstWidget->sizePolicy().hasHeightForWidth());
        OCCInstWidget->setSizePolicy(sizePolicy);
        verticalLayout = new QVBoxLayout(OCCInstWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame = new QFrame(OCCInstWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Plain);
        verticalLayout_2 = new QVBoxLayout(frame);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        b_insert_left = new QPushButton(frame);
        b_insert_left->setObjectName(QString::fromUtf8("b_insert_left"));

        horizontalLayout->addWidget(b_insert_left);

        b_delete_inst = new QPushButton(frame);
        b_delete_inst->setObjectName(QString::fromUtf8("b_delete_inst"));

        horizontalLayout->addWidget(b_delete_inst);

        b_insert_right = new QPushButton(frame);
        b_insert_right->setObjectName(QString::fromUtf8("b_insert_right"));

        horizontalLayout->addWidget(b_insert_right);


        verticalLayout_3->addLayout(horizontalLayout);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        cmb_occ_instructions = new QComboBox(frame);
        cmb_occ_instructions->setObjectName(QString::fromUtf8("cmb_occ_instructions"));

        verticalLayout_3->addWidget(cmb_occ_instructions);

        name_label = new QLabel(frame);
        name_label->setObjectName(QString::fromUtf8("name_label"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(name_label->sizePolicy().hasHeightForWidth());
        name_label->setSizePolicy(sizePolicy1);
        name_label->setAlignment(Qt::AlignCenter);

        verticalLayout_3->addWidget(name_label);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer_2);


        verticalLayout_2->addLayout(verticalLayout_3);


        verticalLayout->addWidget(frame);


        retranslateUi(OCCInstWidget);
        QObject::connect(b_delete_inst, SIGNAL(clicked()), OCCInstWidget, SLOT(b_delete_inst_slot()));
        QObject::connect(b_insert_left, SIGNAL(clicked()), OCCInstWidget, SLOT(b_insert_left_slot()));
        QObject::connect(b_insert_right, SIGNAL(clicked()), OCCInstWidget, SLOT(b_insert_right_slot()));
        QObject::connect(cmb_occ_instructions, SIGNAL(currentTextChanged(QString)), OCCInstWidget, SLOT(cmb_occ_instructions_slot(QString)));

        QMetaObject::connectSlotsByName(OCCInstWidget);
    } // setupUi

    void retranslateUi(QWidget *OCCInstWidget)
    {
        OCCInstWidget->setWindowTitle(QApplication::translate("OCCInstWidget", "Form", nullptr));
        b_insert_left->setText(QApplication::translate("OCCInstWidget", "<= Insert", nullptr));
        b_delete_inst->setText(QApplication::translate("OCCInstWidget", "Delete", nullptr));
        b_insert_right->setText(QApplication::translate("OCCInstWidget", "Insert =>", nullptr));
        name_label->setText(QApplication::translate("OCCInstWidget", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class OCCInstWidget: public Ui_OCCInstWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // OCCINSTRUCTIONWIDGETUI_H
