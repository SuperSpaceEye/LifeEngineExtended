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
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_WorldEvents
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *world_events_viewer_tab;
    QWidget *world_events_tab;
    QVBoxLayout *verticalLayout_2;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_3;
    QFrame *world_events_window;
    QVBoxLayout *verticalLayout_4;
    QWidget *world_events_widget;
    QVBoxLayout *verticalLayout_6;
    QVBoxLayout *world_events_layout;
    QHBoxLayout *horizontalLayout;
    QPushButton *b_apply_events;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QWidget *current_world_events;
    QVBoxLayout *verticalLayout_7;
    QWidget *widget;
    QVBoxLayout *verticalLayout_5;
    QLabel *world_events_status_label;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *b_start_events;
    QPushButton *b_pause_events;
    QPushButton *b_resume_events;
    QPushButton *b_stop_events;
    QWidget *settings_tab;
    QVBoxLayout *verticalLayout_8;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QVBoxLayout *verticalLayout_9;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_10;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    QLineEdit *le_update_world_events_every_n;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_2;
    QLineEdit *le_collect_info_every_n;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *WorldEvents)
    {
        if (WorldEvents->objectName().isEmpty())
            WorldEvents->setObjectName(QString::fromUtf8("WorldEvents"));
        WorldEvents->resize(1055, 595);
        verticalLayout = new QVBoxLayout(WorldEvents);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        world_events_viewer_tab = new QTabWidget(WorldEvents);
        world_events_viewer_tab->setObjectName(QString::fromUtf8("world_events_viewer_tab"));
        world_events_tab = new QWidget();
        world_events_tab->setObjectName(QString::fromUtf8("world_events_tab"));
        verticalLayout_2 = new QVBoxLayout(world_events_tab);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(world_events_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setLineWidth(0);
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
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(world_events_window->sizePolicy().hasHeightForWidth());
        world_events_window->setSizePolicy(sizePolicy);
        world_events_window->setFrameShape(QFrame::StyledPanel);
        world_events_window->setFrameShadow(QFrame::Raised);
        world_events_window->setLineWidth(0);
        verticalLayout_4 = new QVBoxLayout(world_events_window);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        world_events_widget = new QWidget(world_events_window);
        world_events_widget->setObjectName(QString::fromUtf8("world_events_widget"));
        verticalLayout_6 = new QVBoxLayout(world_events_widget);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        world_events_layout = new QVBoxLayout();
        world_events_layout->setSpacing(9);
        world_events_layout->setObjectName(QString::fromUtf8("world_events_layout"));
        world_events_layout->setSizeConstraint(QLayout::SetDefaultConstraint);
        world_events_layout->setContentsMargins(6, 6, 6, 6);

        verticalLayout_6->addLayout(world_events_layout);


        verticalLayout_4->addWidget(world_events_widget);


        verticalLayout_3->addWidget(world_events_window);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_2->addWidget(scrollArea);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        b_apply_events = new QPushButton(world_events_tab);
        b_apply_events->setObjectName(QString::fromUtf8("b_apply_events"));

        horizontalLayout->addWidget(b_apply_events);

        pushButton = new QPushButton(world_events_tab);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);

        pushButton_2 = new QPushButton(world_events_tab);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        horizontalLayout->addWidget(pushButton_2);


        verticalLayout_2->addLayout(horizontalLayout);

        world_events_viewer_tab->addTab(world_events_tab, QString());
        current_world_events = new QWidget();
        current_world_events->setObjectName(QString::fromUtf8("current_world_events"));
        verticalLayout_7 = new QVBoxLayout(current_world_events);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        widget = new QWidget(current_world_events);
        widget->setObjectName(QString::fromUtf8("widget"));
        verticalLayout_5 = new QVBoxLayout(widget);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        world_events_status_label = new QLabel(widget);
        world_events_status_label->setObjectName(QString::fromUtf8("world_events_status_label"));

        verticalLayout_5->addWidget(world_events_status_label);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        b_start_events = new QPushButton(widget);
        b_start_events->setObjectName(QString::fromUtf8("b_start_events"));

        horizontalLayout_2->addWidget(b_start_events);

        b_pause_events = new QPushButton(widget);
        b_pause_events->setObjectName(QString::fromUtf8("b_pause_events"));

        horizontalLayout_2->addWidget(b_pause_events);

        b_resume_events = new QPushButton(widget);
        b_resume_events->setObjectName(QString::fromUtf8("b_resume_events"));

        horizontalLayout_2->addWidget(b_resume_events);

        b_stop_events = new QPushButton(widget);
        b_stop_events->setObjectName(QString::fromUtf8("b_stop_events"));

        horizontalLayout_2->addWidget(b_stop_events);


        verticalLayout_5->addLayout(horizontalLayout_2);


        verticalLayout_7->addWidget(widget);

        world_events_viewer_tab->addTab(current_world_events, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName(QString::fromUtf8("settings_tab"));
        verticalLayout_8 = new QVBoxLayout(settings_tab);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        verticalLayout_8->setContentsMargins(0, 0, 0, 0);
        scrollArea_2 = new QScrollArea(settings_tab);
        scrollArea_2->setObjectName(QString::fromUtf8("scrollArea_2"));
        scrollArea_2->setLineWidth(0);
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QString::fromUtf8("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 1049, 562));
        verticalLayout_9 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        widget_2 = new QWidget(scrollAreaWidgetContents_2);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        verticalLayout_10 = new QVBoxLayout(widget_2);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_10->addItem(verticalSpacer_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label = new QLabel(widget_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_3->addWidget(label);

        le_update_world_events_every_n = new QLineEdit(widget_2);
        le_update_world_events_every_n->setObjectName(QString::fromUtf8("le_update_world_events_every_n"));

        horizontalLayout_3->addWidget(le_update_world_events_every_n);


        verticalLayout_10->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_2 = new QLabel(widget_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_4->addWidget(label_2);

        le_collect_info_every_n = new QLineEdit(widget_2);
        le_collect_info_every_n->setObjectName(QString::fromUtf8("le_collect_info_every_n"));

        horizontalLayout_4->addWidget(le_collect_info_every_n);


        verticalLayout_10->addLayout(horizontalLayout_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_10->addItem(verticalSpacer);


        verticalLayout_9->addWidget(widget_2);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        verticalLayout_8->addWidget(scrollArea_2);

        world_events_viewer_tab->addTab(settings_tab, QString());

        verticalLayout->addWidget(world_events_viewer_tab);


        retranslateUi(WorldEvents);
        QObject::connect(b_apply_events, SIGNAL(clicked()), WorldEvents, SLOT(b_apply_events_slot()));
        QObject::connect(b_pause_events, SIGNAL(clicked()), WorldEvents, SLOT(b_pause_events_slot()));
        QObject::connect(b_resume_events, SIGNAL(clicked()), WorldEvents, SLOT(b_resume_events_slot()));
        QObject::connect(b_start_events, SIGNAL(clicked()), WorldEvents, SLOT(b_start_events_slot()));
        QObject::connect(b_stop_events, SIGNAL(clicked()), WorldEvents, SLOT(b_stop_events_slot()));
        QObject::connect(le_collect_info_every_n, SIGNAL(returnPressed()), WorldEvents, SLOT(le_collect_info_every_n_slot()));
        QObject::connect(le_update_world_events_every_n, SIGNAL(returnPressed()), WorldEvents, SLOT(le_update_world_events_every_n_slot()));

        world_events_viewer_tab->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(WorldEvents);
    } // setupUi

    void retranslateUi(QWidget *WorldEvents)
    {
        WorldEvents->setWindowTitle(QApplication::translate("WorldEvents", "World Events", nullptr));
        b_apply_events->setText(QApplication::translate("WorldEvents", "Apply events", nullptr));
        pushButton->setText(QApplication::translate("WorldEvents", "Load Events", nullptr));
        pushButton_2->setText(QApplication::translate("WorldEvents", "Save Events", nullptr));
        world_events_viewer_tab->setTabText(world_events_viewer_tab->indexOf(world_events_tab), QApplication::translate("WorldEvents", "World Events Editor", nullptr));
        world_events_status_label->setText(QApplication::translate("WorldEvents", "Events stopped.", nullptr));
        b_start_events->setText(QApplication::translate("WorldEvents", "Start Events", nullptr));
        b_pause_events->setText(QApplication::translate("WorldEvents", "Pause Events", nullptr));
        b_resume_events->setText(QApplication::translate("WorldEvents", "Resume events", nullptr));
        b_stop_events->setText(QApplication::translate("WorldEvents", "Stop Events", nullptr));
        world_events_viewer_tab->setTabText(world_events_viewer_tab->indexOf(current_world_events), QApplication::translate("WorldEvents", "Current World Events Viewer", nullptr));
        label->setText(QApplication::translate("WorldEvents", "Update World Events every n ticks", nullptr));
        label_2->setText(QApplication::translate("WorldEvents", "Collect Info every n ticks", nullptr));
        world_events_viewer_tab->setTabText(world_events_viewer_tab->indexOf(settings_tab), QApplication::translate("WorldEvents", "Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class WorldEvents: public Ui_WorldEvents {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WORLDEVENTSUI_H
