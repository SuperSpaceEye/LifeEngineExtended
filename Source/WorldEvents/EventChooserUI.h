/********************************************************************************
** Form generated from reading UI file 'EventChooser.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef EVENTCHOOSERUI_H
#define EVENTCHOOSERUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_NewEventChooser
{
public:
    QVBoxLayout *verticalLayout;
    QPushButton *b_change_value_event;
    QPushButton *b_condition_event;

    void setupUi(QWidget *NewEventChooser)
    {
        if (NewEventChooser->objectName().isEmpty())
            NewEventChooser->setObjectName(QString::fromUtf8("NewEventChooser"));
        NewEventChooser->resize(400, 200);
        verticalLayout = new QVBoxLayout(NewEventChooser);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        b_change_value_event = new QPushButton(NewEventChooser);
        b_change_value_event->setObjectName(QString::fromUtf8("b_change_value_event"));

        verticalLayout->addWidget(b_change_value_event);

        b_condition_event = new QPushButton(NewEventChooser);
        b_condition_event->setObjectName(QString::fromUtf8("b_condition_event"));

        verticalLayout->addWidget(b_condition_event);


        retranslateUi(NewEventChooser);
        QObject::connect(b_change_value_event, SIGNAL(clicked()), NewEventChooser, SLOT(b_change_value_event_slot()));
        QObject::connect(b_condition_event, SIGNAL(clicked()), NewEventChooser, SLOT(b_condition_event_slot()));

        QMetaObject::connectSlotsByName(NewEventChooser);
    } // setupUi

    void retranslateUi(QWidget *NewEventChooser)
    {
        NewEventChooser->setWindowTitle(QApplication::translate("NewEventChooser", "Form", nullptr));
        b_change_value_event->setText(QApplication::translate("NewEventChooser", "Change value event", nullptr));
        b_condition_event->setText(QApplication::translate("NewEventChooser", "Condition event", nullptr));
    } // retranslateUi

};

namespace Ui {
    class NewEventChooser: public Ui_NewEventChooser {};
} // namespace Ui

QT_END_NAMESPACE

#endif // EVENTCHOOSERUI_H
