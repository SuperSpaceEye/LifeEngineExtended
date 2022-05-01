/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef WINDOWUI_H
#define WINDOWUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout_3;
    QGraphicsView *graphicsView;
    QFrame *horizontalFrame;
    QHBoxLayout *horizontalLayout;
    QFrame *frame_2;
    QVBoxLayout *verticalLayout_6;
    QLabel *fps_label;
    QLineEdit *fps_lineedit;
    QLabel *sps_label;
    QLineEdit *sps_lineedit;
    QGridLayout *gridLayout;
    QPushButton *clear_button;
    QPushButton *reset_button;
    QPushButton *pause_button;
    QPushButton *stoprender_button;
    QPushButton *reset_view_button;
    QPushButton *pass_one_tick_button;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *food_rbutton;
    QRadioButton *kill_rbutton;
    QRadioButton *wall_rbutton;
    QFrame *frame_3;
    QVBoxLayout *verticalLayout_4;
    QTabWidget *Tabs;
    QWidget *editor_tab;
    QWidget *world_controls_tab;
    QWidget *evolution_controls_tab;
    QWidget *simulation_settings_tab;
    QWidget *statistics_tab;
    QButtonGroup *buttonGroup;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 800);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(0, 0));
        verticalLayout_2 = new QVBoxLayout(MainWindow);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame = new QFrame(MainWindow);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout_3 = new QVBoxLayout(frame);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        graphicsView = new QGraphicsView(frame);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(graphicsView->sizePolicy().hasHeightForWidth());
        graphicsView->setSizePolicy(sizePolicy1);
        graphicsView->setLayoutDirection(Qt::LeftToRight);
        graphicsView->setFrameShadow(QFrame::Plain);
        graphicsView->setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);

        verticalLayout_3->addWidget(graphicsView);

        horizontalFrame = new QFrame(frame);
        horizontalFrame->setObjectName(QString::fromUtf8("horizontalFrame"));
        horizontalFrame->setFrameShape(QFrame::NoFrame);
        horizontalLayout = new QHBoxLayout(horizontalFrame);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        frame_2 = new QFrame(horizontalFrame);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setFrameShape(QFrame::NoFrame);
        frame_2->setFrameShadow(QFrame::Sunken);
        frame_2->setLineWidth(0);
        verticalLayout_6 = new QVBoxLayout(frame_2);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(4, 4, 4, 0);
        fps_label = new QLabel(frame_2);
        fps_label->setObjectName(QString::fromUtf8("fps_label"));

        verticalLayout_6->addWidget(fps_label);

        fps_lineedit = new QLineEdit(frame_2);
        fps_lineedit->setObjectName(QString::fromUtf8("fps_lineedit"));

        verticalLayout_6->addWidget(fps_lineedit);

        sps_label = new QLabel(frame_2);
        sps_label->setObjectName(QString::fromUtf8("sps_label"));

        verticalLayout_6->addWidget(sps_label);

        sps_lineedit = new QLineEdit(frame_2);
        sps_lineedit->setObjectName(QString::fromUtf8("sps_lineedit"));

        verticalLayout_6->addWidget(sps_lineedit);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        clear_button = new QPushButton(frame_2);
        clear_button->setObjectName(QString::fromUtf8("clear_button"));

        gridLayout->addWidget(clear_button, 0, 2, 1, 1);

        reset_button = new QPushButton(frame_2);
        reset_button->setObjectName(QString::fromUtf8("reset_button"));

        gridLayout->addWidget(reset_button, 0, 3, 1, 1);

        pause_button = new QPushButton(frame_2);
        pause_button->setObjectName(QString::fromUtf8("pause_button"));
        pause_button->setCheckable(true);

        gridLayout->addWidget(pause_button, 0, 0, 1, 1);

        stoprender_button = new QPushButton(frame_2);
        stoprender_button->setObjectName(QString::fromUtf8("stoprender_button"));
        stoprender_button->setCheckable(true);

        gridLayout->addWidget(stoprender_button, 0, 1, 1, 1);

        reset_view_button = new QPushButton(frame_2);
        reset_view_button->setObjectName(QString::fromUtf8("reset_view_button"));

        gridLayout->addWidget(reset_view_button, 1, 0, 1, 1);

        pass_one_tick_button = new QPushButton(frame_2);
        pass_one_tick_button->setObjectName(QString::fromUtf8("pass_one_tick_button"));

        gridLayout->addWidget(pass_one_tick_button, 1, 1, 1, 1);


        verticalLayout_6->addLayout(gridLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        food_rbutton = new QRadioButton(frame_2);
        buttonGroup = new QButtonGroup(MainWindow);
        buttonGroup->setObjectName(QString::fromUtf8("buttonGroup"));
        buttonGroup->setExclusive(true);
        buttonGroup->addButton(food_rbutton);
        food_rbutton->setObjectName(QString::fromUtf8("food_rbutton"));
        food_rbutton->setChecked(true);

        horizontalLayout_3->addWidget(food_rbutton);

        kill_rbutton = new QRadioButton(frame_2);
        buttonGroup->addButton(kill_rbutton);
        kill_rbutton->setObjectName(QString::fromUtf8("kill_rbutton"));

        horizontalLayout_3->addWidget(kill_rbutton);

        wall_rbutton = new QRadioButton(frame_2);
        buttonGroup->addButton(wall_rbutton);
        wall_rbutton->setObjectName(QString::fromUtf8("wall_rbutton"));

        horizontalLayout_3->addWidget(wall_rbutton);


        verticalLayout_6->addLayout(horizontalLayout_3);


        horizontalLayout->addWidget(frame_2, 0, Qt::AlignLeft);

        frame_3 = new QFrame(horizontalFrame);
        frame_3->setObjectName(QString::fromUtf8("frame_3"));
        frame_3->setFrameShape(QFrame::StyledPanel);
        frame_3->setFrameShadow(QFrame::Raised);
        verticalLayout_4 = new QVBoxLayout(frame_3);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        Tabs = new QTabWidget(frame_3);
        Tabs->setObjectName(QString::fromUtf8("Tabs"));
        Tabs->setTabPosition(QTabWidget::North);
        Tabs->setTabShape(QTabWidget::Rounded);
        Tabs->setIconSize(QSize(16, 16));
        Tabs->setElideMode(Qt::ElideNone);
        Tabs->setDocumentMode(true);
        Tabs->setTabsClosable(false);
        Tabs->setMovable(false);
        Tabs->setTabBarAutoHide(true);
        editor_tab = new QWidget();
        editor_tab->setObjectName(QString::fromUtf8("editor_tab"));
        Tabs->addTab(editor_tab, QString());
        world_controls_tab = new QWidget();
        world_controls_tab->setObjectName(QString::fromUtf8("world_controls_tab"));
        Tabs->addTab(world_controls_tab, QString());
        evolution_controls_tab = new QWidget();
        evolution_controls_tab->setObjectName(QString::fromUtf8("evolution_controls_tab"));
        Tabs->addTab(evolution_controls_tab, QString());
        simulation_settings_tab = new QWidget();
        simulation_settings_tab->setObjectName(QString::fromUtf8("simulation_settings_tab"));
        Tabs->addTab(simulation_settings_tab, QString());
        statistics_tab = new QWidget();
        statistics_tab->setObjectName(QString::fromUtf8("statistics_tab"));
        Tabs->addTab(statistics_tab, QString());

        verticalLayout_4->addWidget(Tabs);


        horizontalLayout->addWidget(frame_3);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 2);
        frame_3->raise();
        frame_2->raise();

        verticalLayout_3->addWidget(horizontalFrame);

        verticalLayout_3->setStretch(0, 3);
        verticalLayout_3->setStretch(1, 1);

        verticalLayout->addWidget(frame);


        verticalLayout_2->addLayout(verticalLayout);

#ifndef QT_NO_SHORTCUT
        fps_label->setBuddy(fps_lineedit);
#endif // QT_NO_SHORTCUT

        retranslateUi(MainWindow);
        QObject::connect(pause_button, SIGNAL(toggled(bool)), MainWindow, SLOT(pause_slot(bool)));
        QObject::connect(stoprender_button, SIGNAL(toggled(bool)), MainWindow, SLOT(stoprender_slot(bool)));
        QObject::connect(clear_button, SIGNAL(clicked()), MainWindow, SLOT(clear_slot()));
        QObject::connect(reset_button, SIGNAL(clicked()), MainWindow, SLOT(reset_slot()));
        QObject::connect(reset_view_button, SIGNAL(clicked()), MainWindow, SLOT(reset_view_slot()));
        QObject::connect(pass_one_tick_button, SIGNAL(clicked()), MainWindow, SLOT(pass_one_tick_slot()));

        Tabs->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        fps_label->setText(QApplication::translate("MainWindow", "fps:", nullptr));
        fps_lineedit->setText(QApplication::translate("MainWindow", "set max fps", nullptr));
        sps_label->setText(QApplication::translate("MainWindow", "sps:", nullptr));
        sps_lineedit->setText(QApplication::translate("MainWindow", "set max sps", nullptr));
        clear_button->setText(QApplication::translate("MainWindow", "Clear", nullptr));
        reset_button->setText(QApplication::translate("MainWindow", "Reset", nullptr));
        pause_button->setText(QApplication::translate("MainWindow", "Pause", nullptr));
        stoprender_button->setText(QApplication::translate("MainWindow", "Stop render", nullptr));
        reset_view_button->setText(QApplication::translate("MainWindow", "Reset view", nullptr));
        pass_one_tick_button->setText(QApplication::translate("MainWindow", "Pass one tick", nullptr));
        food_rbutton->setText(QApplication::translate("MainWindow", "Food mode", nullptr));
        kill_rbutton->setText(QApplication::translate("MainWindow", "Kill mode", nullptr));
        wall_rbutton->setText(QApplication::translate("MainWindow", "Wall mode", nullptr));
        Tabs->setTabText(Tabs->indexOf(editor_tab), QApplication::translate("MainWindow", "Editor", nullptr));
        Tabs->setTabText(Tabs->indexOf(world_controls_tab), QApplication::translate("MainWindow", "World Controls", nullptr));
        Tabs->setTabText(Tabs->indexOf(evolution_controls_tab), QApplication::translate("MainWindow", "Evolution Controls", nullptr));
        Tabs->setTabText(Tabs->indexOf(simulation_settings_tab), QApplication::translate("MainWindow", "Simulation Settings", nullptr));
        Tabs->setTabText(Tabs->indexOf(statistics_tab), QApplication::translate("MainWindow", "Statistics", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WINDOWUI_H
