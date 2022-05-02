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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QGraphicsView *graphicsView;
    QFrame *horizontalFrame;
    QHBoxLayout *horizontalLayout;
    QFrame *frame_2;
    QVBoxLayout *verticalLayout_6;
    QLabel *fps_label;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_2;
    QLineEdit *fps_lineedit;
    QLabel *sps_label;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_3;
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
    QWidget *statistics_tab;
    QWidget *simulation_settings_tab;
    QVBoxLayout *verticalLayout_7;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_5;
    QCheckBox *stop_console_output_check;
    QCheckBox *synchronise_sim_and_win_check;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *single_thread_mode_rbutton;
    QRadioButton *multi_thread_mode_rbutton;
    QRadioButton *cuda_mode_rbutton;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label;
    QLineEdit *set_cpu_threads_linedit;
    QButtonGroup *simulation_modes;
    QButtonGroup *cursor_modes;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 900);
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
        graphicsView = new QGraphicsView(MainWindow);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(graphicsView->sizePolicy().hasHeightForWidth());
        graphicsView->setSizePolicy(sizePolicy1);
        graphicsView->setLayoutDirection(Qt::LeftToRight);
        graphicsView->setStyleSheet(QString::fromUtf8("border: 0px;\n"
"border-radius: 0px;"));
        graphicsView->setFrameShape(QFrame::NoFrame);
        graphicsView->setFrameShadow(QFrame::Plain);
        graphicsView->setLineWidth(0);
        graphicsView->setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);

        verticalLayout->addWidget(graphicsView);

        horizontalFrame = new QFrame(MainWindow);
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

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_2 = new QLabel(frame_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_6->addWidget(label_2);

        fps_lineedit = new QLineEdit(frame_2);
        fps_lineedit->setObjectName(QString::fromUtf8("fps_lineedit"));

        horizontalLayout_6->addWidget(fps_lineedit);


        verticalLayout_6->addLayout(horizontalLayout_6);

        sps_label = new QLabel(frame_2);
        sps_label->setObjectName(QString::fromUtf8("sps_label"));

        verticalLayout_6->addWidget(sps_label);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_3 = new QLabel(frame_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_5->addWidget(label_3);

        sps_lineedit = new QLineEdit(frame_2);
        sps_lineedit->setObjectName(QString::fromUtf8("sps_lineedit"));

        horizontalLayout_5->addWidget(sps_lineedit);


        verticalLayout_6->addLayout(horizontalLayout_5);

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
        cursor_modes = new QButtonGroup(MainWindow);
        cursor_modes->setObjectName(QString::fromUtf8("cursor_modes"));
        cursor_modes->setExclusive(true);
        cursor_modes->addButton(food_rbutton);
        food_rbutton->setObjectName(QString::fromUtf8("food_rbutton"));
        food_rbutton->setChecked(true);

        horizontalLayout_3->addWidget(food_rbutton);

        kill_rbutton = new QRadioButton(frame_2);
        cursor_modes->addButton(kill_rbutton);
        kill_rbutton->setObjectName(QString::fromUtf8("kill_rbutton"));

        horizontalLayout_3->addWidget(kill_rbutton);

        wall_rbutton = new QRadioButton(frame_2);
        cursor_modes->addButton(wall_rbutton);
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
        statistics_tab = new QWidget();
        statistics_tab->setObjectName(QString::fromUtf8("statistics_tab"));
        Tabs->addTab(statistics_tab, QString());
        simulation_settings_tab = new QWidget();
        simulation_settings_tab->setObjectName(QString::fromUtf8("simulation_settings_tab"));
        verticalLayout_7 = new QVBoxLayout(simulation_settings_tab);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        scrollArea = new QScrollArea(simulation_settings_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 410, 248));
        verticalLayout_5 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        stop_console_output_check = new QCheckBox(scrollAreaWidgetContents);
        stop_console_output_check->setObjectName(QString::fromUtf8("stop_console_output_check"));

        verticalLayout_5->addWidget(stop_console_output_check);

        synchronise_sim_and_win_check = new QCheckBox(scrollAreaWidgetContents);
        synchronise_sim_and_win_check->setObjectName(QString::fromUtf8("synchronise_sim_and_win_check"));
        synchronise_sim_and_win_check->setChecked(false);

        verticalLayout_5->addWidget(synchronise_sim_and_win_check);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        single_thread_mode_rbutton = new QRadioButton(scrollAreaWidgetContents);
        simulation_modes = new QButtonGroup(MainWindow);
        simulation_modes->setObjectName(QString::fromUtf8("simulation_modes"));
        simulation_modes->addButton(single_thread_mode_rbutton);
        single_thread_mode_rbutton->setObjectName(QString::fromUtf8("single_thread_mode_rbutton"));

        horizontalLayout_2->addWidget(single_thread_mode_rbutton);

        multi_thread_mode_rbutton = new QRadioButton(scrollAreaWidgetContents);
        simulation_modes->addButton(multi_thread_mode_rbutton);
        multi_thread_mode_rbutton->setObjectName(QString::fromUtf8("multi_thread_mode_rbutton"));
        multi_thread_mode_rbutton->setChecked(true);

        horizontalLayout_2->addWidget(multi_thread_mode_rbutton);

        cuda_mode_rbutton = new QRadioButton(scrollAreaWidgetContents);
        simulation_modes->addButton(cuda_mode_rbutton);
        cuda_mode_rbutton->setObjectName(QString::fromUtf8("cuda_mode_rbutton"));

        horizontalLayout_2->addWidget(cuda_mode_rbutton);


        verticalLayout_5->addLayout(horizontalLayout_2);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label = new QLabel(scrollAreaWidgetContents);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_4->addWidget(label);

        set_cpu_threads_linedit = new QLineEdit(scrollAreaWidgetContents);
        set_cpu_threads_linedit->setObjectName(QString::fromUtf8("set_cpu_threads_linedit"));

        horizontalLayout_4->addWidget(set_cpu_threads_linedit);


        verticalLayout_5->addLayout(horizontalLayout_4);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_7->addWidget(scrollArea);

        Tabs->addTab(simulation_settings_tab, QString());

        verticalLayout_4->addWidget(Tabs);


        horizontalLayout->addWidget(frame_3);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 2);
        frame_3->raise();
        frame_2->raise();

        verticalLayout->addWidget(horizontalFrame);

        verticalLayout->setStretch(0, 2);
        verticalLayout->setStretch(1, 1);

        verticalLayout_2->addLayout(verticalLayout);

#ifndef QT_NO_SHORTCUT
        fps_label->setBuddy(fps_lineedit);
#endif // QT_NO_SHORTCUT

        retranslateUi(MainWindow);
        QObject::connect(reset_button, SIGNAL(clicked()), MainWindow, SLOT(reset_slot()));
        QObject::connect(food_rbutton, SIGNAL(clicked()), MainWindow, SLOT(food_rbutton_slot()));
        QObject::connect(stop_console_output_check, SIGNAL(toggled(bool)), MainWindow, SLOT(stop_console_output_slot(bool)));
        QObject::connect(synchronise_sim_and_win_check, SIGNAL(toggled(bool)), MainWindow, SLOT(synchronise_simulation_and_window_slot(bool)));
        QObject::connect(stoprender_button, SIGNAL(toggled(bool)), MainWindow, SLOT(stoprender_slot(bool)));
        QObject::connect(multi_thread_mode_rbutton, SIGNAL(clicked()), MainWindow, SLOT(multi_thread_rbutton_slot()));
        QObject::connect(cuda_mode_rbutton, SIGNAL(clicked()), MainWindow, SLOT(cuda_rbutton_slot()));
        QObject::connect(wall_rbutton, SIGNAL(clicked()), MainWindow, SLOT(wall_rbutton_slot()));
        QObject::connect(reset_view_button, SIGNAL(clicked()), MainWindow, SLOT(reset_view_slot()));
        QObject::connect(kill_rbutton, SIGNAL(clicked()), MainWindow, SLOT(kill_rbutton_slot()));
        QObject::connect(fps_lineedit, SIGNAL(editingFinished()), MainWindow, SLOT(parse_max_fps_slot()));
        QObject::connect(clear_button, SIGNAL(clicked()), MainWindow, SLOT(clear_slot()));
        QObject::connect(single_thread_mode_rbutton, SIGNAL(clicked()), MainWindow, SLOT(single_thread_rbutton_slot()));
        QObject::connect(pause_button, SIGNAL(toggled(bool)), MainWindow, SLOT(pause_slot(bool)));
        QObject::connect(set_cpu_threads_linedit, SIGNAL(returnPressed()), MainWindow, SLOT(parse_num_threads_slot()));
        QObject::connect(sps_lineedit, SIGNAL(editingFinished()), MainWindow, SLOT(parse_max_sps_slot()));
        QObject::connect(pass_one_tick_button, SIGNAL(clicked()), MainWindow, SLOT(pass_one_tick_slot()));

        Tabs->setCurrentIndex(4);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        fps_label->setText(QApplication::translate("MainWindow", "fps:", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "Set max fps:", nullptr));
        fps_lineedit->setText(QApplication::translate("MainWindow", "60", nullptr));
        sps_label->setText(QApplication::translate("MainWindow", "sps:", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "Set max sps:", nullptr));
        sps_lineedit->setText(QApplication::translate("MainWindow", "-1", nullptr));
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
        Tabs->setTabText(Tabs->indexOf(statistics_tab), QApplication::translate("MainWindow", "Statistics", nullptr));
        stop_console_output_check->setText(QApplication::translate("MainWindow", "Stop console output", nullptr));
        synchronise_sim_and_win_check->setText(QApplication::translate("MainWindow", "Synchronise simulation and window", nullptr));
        single_thread_mode_rbutton->setText(QApplication::translate("MainWindow", "Single thread CPU", nullptr));
        multi_thread_mode_rbutton->setText(QApplication::translate("MainWindow", "Multi-thread CPU", nullptr));
        cuda_mode_rbutton->setText(QApplication::translate("MainWindow", "CUDA", nullptr));
        label->setText(QApplication::translate("MainWindow", "Set number of CPU threads:", nullptr));
        set_cpu_threads_linedit->setText(QApplication::translate("MainWindow", "1", nullptr));
        Tabs->setTabText(Tabs->indexOf(simulation_settings_tab), QApplication::translate("MainWindow", "Simulation Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WINDOWUI_H
