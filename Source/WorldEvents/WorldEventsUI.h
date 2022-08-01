/********************************************************************************
** Form generated from reading UI file 'worldevents.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef WORLDEVENTSUI_H
#define WORLDEVENTSUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_WorldEvents
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *world_events_tab;
    QVBoxLayout *verticalLayout_2;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_3;
    QFrame *world_events_window;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QWidget *settings_tab;

    void setupUi(QWidget *WorldEvents)
    {
        if (WorldEvents->objectName().isEmpty())
            WorldEvents->setObjectName(QString::fromUtf8("WorldEvents"));
        WorldEvents->resize(1055, 595);
        verticalLayout = new QVBoxLayout(WorldEvents);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(WorldEvents);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        world_events_tab = new QWidget();
        world_events_tab->setObjectName(QString::fromUtf8("world_events_tab"));
        verticalLayout_2 = new QVBoxLayout(world_events_tab);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(world_events_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1049, 535));
        verticalLayout_3 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        world_events_window = new QFrame(scrollAreaWidgetContents);
        world_events_window->setObjectName(QString::fromUtf8("world_events_window"));
        world_events_window->setFrameShape(QFrame::StyledPanel);
        world_events_window->setFrameShadow(QFrame::Raised);

        verticalLayout_3->addWidget(world_events_window);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_2->addWidget(scrollArea);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        pushButton = new QPushButton(world_events_tab);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);

        pushButton_2 = new QPushButton(world_events_tab);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        horizontalLayout->addWidget(pushButton_2);


        verticalLayout_2->addLayout(horizontalLayout);

        tabWidget->addTab(world_events_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName(QString::fromUtf8("settings_tab"));
        tabWidget->addTab(settings_tab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(WorldEvents);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(WorldEvents);
    } // setupUi

    void retranslateUi(QWidget *WorldEvents)
    {
        WorldEvents->setWindowTitle(QApplication::translate("WorldEvents", "World Events", nullptr));
        pushButton->setText(QApplication::translate("WorldEvents", "Load Events", nullptr));
        pushButton_2->setText(QApplication::translate("WorldEvents", "Save Events", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(world_events_tab), QApplication::translate("WorldEvents", "World Events Editor", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(settings_tab), QApplication::translate("WorldEvents", "Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class WorldEvents: public Ui_WorldEvents {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WORLDEVENTSUI_H
