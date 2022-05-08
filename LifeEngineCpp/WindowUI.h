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
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QVBoxLayout *verticalLayout_2;
    QGraphicsView *simulation_graphicsView;
    QFrame *horizontalFrame;
    QHBoxLayout *horizontalLayout;
    QFrame *frame_2;
    QVBoxLayout *verticalLayout_6;
    QLabel *lb_fps;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_2;
    QLineEdit *le_fps;
    QLabel *lb_sps;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_3;
    QLineEdit *le_sps;
    QGridLayout *gridLayout;
    QPushButton *b_clear;
    QPushButton *b_reset;
    QPushButton *tb_pause;
    QPushButton *tb_stoprender;
    QPushButton *b_reset_view;
    QPushButton *b_pass_one_tick;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *rb_food;
    QRadioButton *rb_kill;
    QRadioButton *rb_wall;
    QFrame *frame_3;
    QVBoxLayout *verticalLayout_4;
    QTabWidget *Tabs;
    QWidget *about_tab;
    QVBoxLayout *verticalLayout_8;
    QTextEdit *textEdit;
    QWidget *editor_tab;
    QWidget *world_controls_tab;
    QVBoxLayout *verticalLayout;
    QScrollArea *scrollArea_3;
    QWidget *scrollAreaWidgetContents_3;
    QVBoxLayout *verticalLayout_12;
    QWidget *widget_3;
    QHBoxLayout *horizontalLayout_20;
    QVBoxLayout *verticalLayout_21;
    QHBoxLayout *horizontalLayout_22;
    QLabel *label_8;
    QLineEdit *le_cell_size;
    QCheckBox *cb_fill_window;
    QHBoxLayout *horizontalLayout_23;
    QLabel *label_13;
    QLineEdit *le_simulation_width;
    QLabel *label_14;
    QLineEdit *le_simulation_height;
    QPushButton *b_resize_and_reset;
    QCheckBox *cb_reset_on_total_extinction;
    QLabel *lb_auto_reset_count;
    QCheckBox *cb_pause_on_total_extinction;
    QVBoxLayout *verticalLayout_22;
    QPushButton *b_generate_random_walls;
    QCheckBox *cb_generate_random_walls_on_reset;
    QPushButton *b_clear_all_walls;
    QCheckBox *cb_clear_walls_on_reset;
    QPushButton *b_save_world;
    QHBoxLayout *horizontalLayout_21;
    QPushButton *b_load_world;
    QCheckBox *cb_override_evolution_controls;
    QWidget *evolution_controls_tab;
    QHBoxLayout *horizontalLayout_7;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QVBoxLayout *verticalLayout_9;
    QWidget *widget;
    QVBoxLayout *verticalLayout_20;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_4;
    QLineEdit *le_food_production_probability;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_5;
    QLineEdit *le_lifespan_multiplier;
    QVBoxLayout *verticalLayout_18;
    QHBoxLayout *horizontalLayout_13;
    QLabel *label_7;
    QLineEdit *le_look_range;
    QVBoxLayout *verticalLayout_14;
    QHBoxLayout *horizontalLayout_15;
    QLabel *label_9;
    QLineEdit *le_auto_food_drop_rate;
    QHBoxLayout *horizontalLayout_18;
    QLabel *label_6;
    QLineEdit *le_extra_reproduction_cost;
    QVBoxLayout *verticalLayout_16;
    QHBoxLayout *horizontalLayout_19;
    QCheckBox *cb_use_evoved_mutation_rate;
    QVBoxLayout *mutation_rate_layout;
    QHBoxLayout *horizontalLayout_12;
    QLabel *lb_mutation_rate;
    QLineEdit *le_global_mutation_rate;
    QVBoxLayout *verticalLayout_17;
    QHBoxLayout *horizontalLayout_16;
    QLabel *label_10;
    QLineEdit *le_add;
    QLabel *label_11;
    QLineEdit *le_change;
    QLabel *label_12;
    QLineEdit *le_remove;
    QVBoxLayout *verticalLayout_11;
    QHBoxLayout *horizontalLayout_11;
    QCheckBox *cb_rotation_enabled;
    QVBoxLayout *verticalLayout_10;
    QHBoxLayout *horizontalLayout_10;
    QCheckBox *cb_on_touch_kill;
    QVBoxLayout *verticalLayout_15;
    QHBoxLayout *horizontalLayout_17;
    QCheckBox *cb_movers_can_produce_food;
    QVBoxLayout *verticalLayout_13;
    QHBoxLayout *horizontalLayout_14;
    QCheckBox *cb_food_blocks_reproduction;
    QWidget *statistics_tab;
    QWidget *simulation_settings_tab;
    QVBoxLayout *verticalLayout_7;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_5;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_19;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *rb_single_thread_mode;
    QRadioButton *rb_multi_thread_mode;
    QRadioButton *rb_cuda_mode;
    QCheckBox *cb_stop_console_output;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label;
    QLineEdit *le_num_threads;
    QCheckBox *cb_synchronise_sim_and_win;
    QButtonGroup *simulation_modes;
    QButtonGroup *cursor_modes;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 900);
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(0, 0));
        verticalLayout_2 = new QVBoxLayout(MainWindow);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setSizeConstraint(QLayout::SetMaximumSize);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        simulation_graphicsView = new QGraphicsView(MainWindow);
        simulation_graphicsView->setObjectName(QString::fromUtf8("simulation_graphicsView"));
        sizePolicy.setHeightForWidth(simulation_graphicsView->sizePolicy().hasHeightForWidth());
        simulation_graphicsView->setSizePolicy(sizePolicy);
        simulation_graphicsView->setLayoutDirection(Qt::LeftToRight);
        simulation_graphicsView->setStyleSheet(QString::fromUtf8(""));
        simulation_graphicsView->setFrameShape(QFrame::NoFrame);
        simulation_graphicsView->setFrameShadow(QFrame::Plain);
        simulation_graphicsView->setLineWidth(0);
        simulation_graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        simulation_graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        simulation_graphicsView->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
        simulation_graphicsView->setInteractive(false);
        simulation_graphicsView->setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);

        verticalLayout_2->addWidget(simulation_graphicsView);

        horizontalFrame = new QFrame(MainWindow);
        horizontalFrame->setObjectName(QString::fromUtf8("horizontalFrame"));
        sizePolicy.setHeightForWidth(horizontalFrame->sizePolicy().hasHeightForWidth());
        horizontalFrame->setSizePolicy(sizePolicy);
        horizontalFrame->setFrameShape(QFrame::NoFrame);
        horizontalLayout = new QHBoxLayout(horizontalFrame);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        frame_2 = new QFrame(horizontalFrame);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        sizePolicy.setHeightForWidth(frame_2->sizePolicy().hasHeightForWidth());
        frame_2->setSizePolicy(sizePolicy);
        frame_2->setFrameShape(QFrame::NoFrame);
        frame_2->setFrameShadow(QFrame::Sunken);
        frame_2->setLineWidth(0);
        verticalLayout_6 = new QVBoxLayout(frame_2);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(9, 9, 9, 9);
        lb_fps = new QLabel(frame_2);
        lb_fps->setObjectName(QString::fromUtf8("lb_fps"));

        verticalLayout_6->addWidget(lb_fps);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_2 = new QLabel(frame_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_6->addWidget(label_2);

        le_fps = new QLineEdit(frame_2);
        le_fps->setObjectName(QString::fromUtf8("le_fps"));

        horizontalLayout_6->addWidget(le_fps);


        verticalLayout_6->addLayout(horizontalLayout_6);

        lb_sps = new QLabel(frame_2);
        lb_sps->setObjectName(QString::fromUtf8("lb_sps"));

        verticalLayout_6->addWidget(lb_sps);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_3 = new QLabel(frame_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_5->addWidget(label_3);

        le_sps = new QLineEdit(frame_2);
        le_sps->setObjectName(QString::fromUtf8("le_sps"));

        horizontalLayout_5->addWidget(le_sps);


        verticalLayout_6->addLayout(horizontalLayout_5);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        b_clear = new QPushButton(frame_2);
        b_clear->setObjectName(QString::fromUtf8("b_clear"));

        gridLayout->addWidget(b_clear, 0, 2, 1, 1);

        b_reset = new QPushButton(frame_2);
        b_reset->setObjectName(QString::fromUtf8("b_reset"));

        gridLayout->addWidget(b_reset, 0, 3, 1, 1);

        tb_pause = new QPushButton(frame_2);
        tb_pause->setObjectName(QString::fromUtf8("tb_pause"));
        tb_pause->setCheckable(true);

        gridLayout->addWidget(tb_pause, 0, 0, 1, 1);

        tb_stoprender = new QPushButton(frame_2);
        tb_stoprender->setObjectName(QString::fromUtf8("tb_stoprender"));
        tb_stoprender->setCheckable(true);

        gridLayout->addWidget(tb_stoprender, 0, 1, 1, 1);

        b_reset_view = new QPushButton(frame_2);
        b_reset_view->setObjectName(QString::fromUtf8("b_reset_view"));

        gridLayout->addWidget(b_reset_view, 1, 0, 1, 1);

        b_pass_one_tick = new QPushButton(frame_2);
        b_pass_one_tick->setObjectName(QString::fromUtf8("b_pass_one_tick"));

        gridLayout->addWidget(b_pass_one_tick, 1, 1, 1, 1);


        verticalLayout_6->addLayout(gridLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        rb_food = new QRadioButton(frame_2);
        cursor_modes = new QButtonGroup(MainWindow);
        cursor_modes->setObjectName(QString::fromUtf8("cursor_modes"));
        cursor_modes->setExclusive(true);
        cursor_modes->addButton(rb_food);
        rb_food->setObjectName(QString::fromUtf8("rb_food"));
        rb_food->setChecked(true);

        horizontalLayout_3->addWidget(rb_food);

        rb_kill = new QRadioButton(frame_2);
        cursor_modes->addButton(rb_kill);
        rb_kill->setObjectName(QString::fromUtf8("rb_kill"));

        horizontalLayout_3->addWidget(rb_kill);

        rb_wall = new QRadioButton(frame_2);
        cursor_modes->addButton(rb_wall);
        rb_wall->setObjectName(QString::fromUtf8("rb_wall"));

        horizontalLayout_3->addWidget(rb_wall);


        verticalLayout_6->addLayout(horizontalLayout_3);


        horizontalLayout->addWidget(frame_2);

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
        about_tab = new QWidget();
        about_tab->setObjectName(QString::fromUtf8("about_tab"));
        verticalLayout_8 = new QVBoxLayout(about_tab);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        textEdit = new QTextEdit(about_tab);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));
        textEdit->setReadOnly(true);
        textEdit->setOverwriteMode(false);

        verticalLayout_8->addWidget(textEdit);

        Tabs->addTab(about_tab, QString());
        editor_tab = new QWidget();
        editor_tab->setObjectName(QString::fromUtf8("editor_tab"));
        Tabs->addTab(editor_tab, QString());
        world_controls_tab = new QWidget();
        world_controls_tab->setObjectName(QString::fromUtf8("world_controls_tab"));
        verticalLayout = new QVBoxLayout(world_controls_tab);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea_3 = new QScrollArea(world_controls_tab);
        scrollArea_3->setObjectName(QString::fromUtf8("scrollArea_3"));
        scrollArea_3->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName(QString::fromUtf8("scrollAreaWidgetContents_3"));
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 420, 192));
        QSizePolicy sizePolicy1(QSizePolicy::Ignored, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(scrollAreaWidgetContents_3->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_3->setSizePolicy(sizePolicy1);
        verticalLayout_12 = new QVBoxLayout(scrollAreaWidgetContents_3);
        verticalLayout_12->setSpacing(0);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        verticalLayout_12->setContentsMargins(0, 0, 0, 0);
        widget_3 = new QWidget(scrollAreaWidgetContents_3);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        horizontalLayout_20 = new QHBoxLayout(widget_3);
        horizontalLayout_20->setObjectName(QString::fromUtf8("horizontalLayout_20"));
        horizontalLayout_20->setContentsMargins(6, 6, 6, 6);
        verticalLayout_21 = new QVBoxLayout();
        verticalLayout_21->setObjectName(QString::fromUtf8("verticalLayout_21"));
        horizontalLayout_22 = new QHBoxLayout();
        horizontalLayout_22->setObjectName(QString::fromUtf8("horizontalLayout_22"));
        label_8 = new QLabel(widget_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_22->addWidget(label_8);

        le_cell_size = new QLineEdit(widget_3);
        le_cell_size->setObjectName(QString::fromUtf8("le_cell_size"));

        horizontalLayout_22->addWidget(le_cell_size);

        cb_fill_window = new QCheckBox(widget_3);
        cb_fill_window->setObjectName(QString::fromUtf8("cb_fill_window"));

        horizontalLayout_22->addWidget(cb_fill_window);


        verticalLayout_21->addLayout(horizontalLayout_22);

        horizontalLayout_23 = new QHBoxLayout();
        horizontalLayout_23->setObjectName(QString::fromUtf8("horizontalLayout_23"));
        label_13 = new QLabel(widget_3);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_23->addWidget(label_13);

        le_simulation_width = new QLineEdit(widget_3);
        le_simulation_width->setObjectName(QString::fromUtf8("le_simulation_width"));

        horizontalLayout_23->addWidget(le_simulation_width);

        label_14 = new QLabel(widget_3);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        horizontalLayout_23->addWidget(label_14);

        le_simulation_height = new QLineEdit(widget_3);
        le_simulation_height->setObjectName(QString::fromUtf8("le_simulation_height"));

        horizontalLayout_23->addWidget(le_simulation_height);


        verticalLayout_21->addLayout(horizontalLayout_23);

        b_resize_and_reset = new QPushButton(widget_3);
        b_resize_and_reset->setObjectName(QString::fromUtf8("b_resize_and_reset"));

        verticalLayout_21->addWidget(b_resize_and_reset);

        cb_reset_on_total_extinction = new QCheckBox(widget_3);
        cb_reset_on_total_extinction->setObjectName(QString::fromUtf8("cb_reset_on_total_extinction"));

        verticalLayout_21->addWidget(cb_reset_on_total_extinction);

        lb_auto_reset_count = new QLabel(widget_3);
        lb_auto_reset_count->setObjectName(QString::fromUtf8("lb_auto_reset_count"));

        verticalLayout_21->addWidget(lb_auto_reset_count);

        cb_pause_on_total_extinction = new QCheckBox(widget_3);
        cb_pause_on_total_extinction->setObjectName(QString::fromUtf8("cb_pause_on_total_extinction"));

        verticalLayout_21->addWidget(cb_pause_on_total_extinction);


        horizontalLayout_20->addLayout(verticalLayout_21);

        verticalLayout_22 = new QVBoxLayout();
        verticalLayout_22->setObjectName(QString::fromUtf8("verticalLayout_22"));
        b_generate_random_walls = new QPushButton(widget_3);
        b_generate_random_walls->setObjectName(QString::fromUtf8("b_generate_random_walls"));

        verticalLayout_22->addWidget(b_generate_random_walls);

        cb_generate_random_walls_on_reset = new QCheckBox(widget_3);
        cb_generate_random_walls_on_reset->setObjectName(QString::fromUtf8("cb_generate_random_walls_on_reset"));

        verticalLayout_22->addWidget(cb_generate_random_walls_on_reset);

        b_clear_all_walls = new QPushButton(widget_3);
        b_clear_all_walls->setObjectName(QString::fromUtf8("b_clear_all_walls"));

        verticalLayout_22->addWidget(b_clear_all_walls);

        cb_clear_walls_on_reset = new QCheckBox(widget_3);
        cb_clear_walls_on_reset->setObjectName(QString::fromUtf8("cb_clear_walls_on_reset"));

        verticalLayout_22->addWidget(cb_clear_walls_on_reset);

        b_save_world = new QPushButton(widget_3);
        b_save_world->setObjectName(QString::fromUtf8("b_save_world"));

        verticalLayout_22->addWidget(b_save_world);

        horizontalLayout_21 = new QHBoxLayout();
        horizontalLayout_21->setObjectName(QString::fromUtf8("horizontalLayout_21"));
        b_load_world = new QPushButton(widget_3);
        b_load_world->setObjectName(QString::fromUtf8("b_load_world"));

        horizontalLayout_21->addWidget(b_load_world);

        cb_override_evolution_controls = new QCheckBox(widget_3);
        cb_override_evolution_controls->setObjectName(QString::fromUtf8("cb_override_evolution_controls"));

        horizontalLayout_21->addWidget(cb_override_evolution_controls);


        verticalLayout_22->addLayout(horizontalLayout_21);


        horizontalLayout_20->addLayout(verticalLayout_22);

        horizontalLayout_20->setStretch(0, 1);
        horizontalLayout_20->setStretch(1, 1);

        verticalLayout_12->addWidget(widget_3);

        scrollArea_3->setWidget(scrollAreaWidgetContents_3);

        verticalLayout->addWidget(scrollArea_3);

        Tabs->addTab(world_controls_tab, QString());
        evolution_controls_tab = new QWidget();
        evolution_controls_tab->setObjectName(QString::fromUtf8("evolution_controls_tab"));
        horizontalLayout_7 = new QHBoxLayout(evolution_controls_tab);
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        scrollArea_2 = new QScrollArea(evolution_controls_tab);
        scrollArea_2->setObjectName(QString::fromUtf8("scrollArea_2"));
        sizePolicy.setHeightForWidth(scrollArea_2->sizePolicy().hasHeightForWidth());
        scrollArea_2->setSizePolicy(sizePolicy);
        scrollArea_2->setMinimumSize(QSize(0, 0));
        scrollArea_2->setLayoutDirection(Qt::LeftToRight);
        scrollArea_2->setAutoFillBackground(false);
        scrollArea_2->setFrameShape(QFrame::StyledPanel);
        scrollArea_2->setLineWidth(-3);
        scrollArea_2->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea_2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea_2->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        scrollArea_2->setWidgetResizable(true);
        scrollArea_2->setAlignment(Qt::AlignCenter);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QString::fromUtf8("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 406, 416));
        sizePolicy1.setHeightForWidth(scrollAreaWidgetContents_2->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_2->setSizePolicy(sizePolicy1);
        verticalLayout_9 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(9, 9, 9, 9);
        widget = new QWidget(scrollAreaWidgetContents_2);
        widget->setObjectName(QString::fromUtf8("widget"));
        QSizePolicy sizePolicy2(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(10);
        sizePolicy2.setVerticalStretch(10);
        sizePolicy2.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy2);
        widget->setMinimumSize(QSize(0, 0));
        verticalLayout_20 = new QVBoxLayout(widget);
        verticalLayout_20->setSpacing(6);
        verticalLayout_20->setObjectName(QString::fromUtf8("verticalLayout_20"));
        verticalLayout_20->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout_20->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_4 = new QLabel(widget);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(label_4->sizePolicy().hasHeightForWidth());
        label_4->setSizePolicy(sizePolicy3);
        label_4->setMinimumSize(QSize(0, 0));
        label_4->setMaximumSize(QSize(16777215, 1666666));

        horizontalLayout_8->addWidget(label_4);

        le_food_production_probability = new QLineEdit(widget);
        le_food_production_probability->setObjectName(QString::fromUtf8("le_food_production_probability"));

        horizontalLayout_8->addWidget(le_food_production_probability);


        verticalLayout_20->addLayout(horizontalLayout_8);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_5 = new QLabel(widget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_9->addWidget(label_5);

        le_lifespan_multiplier = new QLineEdit(widget);
        le_lifespan_multiplier->setObjectName(QString::fromUtf8("le_lifespan_multiplier"));

        horizontalLayout_9->addWidget(le_lifespan_multiplier);


        verticalLayout_20->addLayout(horizontalLayout_9);

        verticalLayout_18 = new QVBoxLayout();
        verticalLayout_18->setObjectName(QString::fromUtf8("verticalLayout_18"));
        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        label_7 = new QLabel(widget);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_13->addWidget(label_7);

        le_look_range = new QLineEdit(widget);
        le_look_range->setObjectName(QString::fromUtf8("le_look_range"));

        horizontalLayout_13->addWidget(le_look_range);


        verticalLayout_18->addLayout(horizontalLayout_13);


        verticalLayout_20->addLayout(verticalLayout_18);

        verticalLayout_14 = new QVBoxLayout();
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        label_9 = new QLabel(widget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout_15->addWidget(label_9);

        le_auto_food_drop_rate = new QLineEdit(widget);
        le_auto_food_drop_rate->setObjectName(QString::fromUtf8("le_auto_food_drop_rate"));

        horizontalLayout_15->addWidget(le_auto_food_drop_rate);


        verticalLayout_14->addLayout(horizontalLayout_15);


        verticalLayout_20->addLayout(verticalLayout_14);

        horizontalLayout_18 = new QHBoxLayout();
        horizontalLayout_18->setObjectName(QString::fromUtf8("horizontalLayout_18"));
        label_6 = new QLabel(widget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_18->addWidget(label_6);

        le_extra_reproduction_cost = new QLineEdit(widget);
        le_extra_reproduction_cost->setObjectName(QString::fromUtf8("le_extra_reproduction_cost"));

        horizontalLayout_18->addWidget(le_extra_reproduction_cost);


        verticalLayout_20->addLayout(horizontalLayout_18);

        verticalLayout_16 = new QVBoxLayout();
        verticalLayout_16->setObjectName(QString::fromUtf8("verticalLayout_16"));
        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setObjectName(QString::fromUtf8("horizontalLayout_19"));
        cb_use_evoved_mutation_rate = new QCheckBox(widget);
        cb_use_evoved_mutation_rate->setObjectName(QString::fromUtf8("cb_use_evoved_mutation_rate"));
        cb_use_evoved_mutation_rate->setChecked(true);

        horizontalLayout_19->addWidget(cb_use_evoved_mutation_rate);


        verticalLayout_16->addLayout(horizontalLayout_19);


        verticalLayout_20->addLayout(verticalLayout_16);

        mutation_rate_layout = new QVBoxLayout();
        mutation_rate_layout->setObjectName(QString::fromUtf8("mutation_rate_layout"));
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        lb_mutation_rate = new QLabel(widget);
        lb_mutation_rate->setObjectName(QString::fromUtf8("lb_mutation_rate"));
        lb_mutation_rate->setEnabled(false);

        horizontalLayout_12->addWidget(lb_mutation_rate);

        le_global_mutation_rate = new QLineEdit(widget);
        le_global_mutation_rate->setObjectName(QString::fromUtf8("le_global_mutation_rate"));
        le_global_mutation_rate->setEnabled(false);

        horizontalLayout_12->addWidget(le_global_mutation_rate);


        mutation_rate_layout->addLayout(horizontalLayout_12);


        verticalLayout_20->addLayout(mutation_rate_layout);

        verticalLayout_17 = new QVBoxLayout();
        verticalLayout_17->setObjectName(QString::fromUtf8("verticalLayout_17"));
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        label_10 = new QLabel(widget);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_16->addWidget(label_10);

        le_add = new QLineEdit(widget);
        le_add->setObjectName(QString::fromUtf8("le_add"));

        horizontalLayout_16->addWidget(le_add);

        label_11 = new QLabel(widget);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        horizontalLayout_16->addWidget(label_11);

        le_change = new QLineEdit(widget);
        le_change->setObjectName(QString::fromUtf8("le_change"));

        horizontalLayout_16->addWidget(le_change);

        label_12 = new QLabel(widget);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        horizontalLayout_16->addWidget(label_12);

        le_remove = new QLineEdit(widget);
        le_remove->setObjectName(QString::fromUtf8("le_remove"));

        horizontalLayout_16->addWidget(le_remove);


        verticalLayout_17->addLayout(horizontalLayout_16);


        verticalLayout_20->addLayout(verticalLayout_17);

        verticalLayout_11 = new QVBoxLayout();
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        cb_rotation_enabled = new QCheckBox(widget);
        cb_rotation_enabled->setObjectName(QString::fromUtf8("cb_rotation_enabled"));

        horizontalLayout_11->addWidget(cb_rotation_enabled);


        verticalLayout_11->addLayout(horizontalLayout_11);


        verticalLayout_20->addLayout(verticalLayout_11);

        verticalLayout_10 = new QVBoxLayout();
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        cb_on_touch_kill = new QCheckBox(widget);
        cb_on_touch_kill->setObjectName(QString::fromUtf8("cb_on_touch_kill"));
        cb_on_touch_kill->setChecked(true);

        horizontalLayout_10->addWidget(cb_on_touch_kill);


        verticalLayout_10->addLayout(horizontalLayout_10);


        verticalLayout_20->addLayout(verticalLayout_10);

        verticalLayout_15 = new QVBoxLayout();
        verticalLayout_15->setObjectName(QString::fromUtf8("verticalLayout_15"));
        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        cb_movers_can_produce_food = new QCheckBox(widget);
        cb_movers_can_produce_food->setObjectName(QString::fromUtf8("cb_movers_can_produce_food"));

        horizontalLayout_17->addWidget(cb_movers_can_produce_food);


        verticalLayout_15->addLayout(horizontalLayout_17);


        verticalLayout_20->addLayout(verticalLayout_15);

        verticalLayout_13 = new QVBoxLayout();
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        cb_food_blocks_reproduction = new QCheckBox(widget);
        cb_food_blocks_reproduction->setObjectName(QString::fromUtf8("cb_food_blocks_reproduction"));
        cb_food_blocks_reproduction->setChecked(true);

        horizontalLayout_14->addWidget(cb_food_blocks_reproduction);


        verticalLayout_13->addLayout(horizontalLayout_14);


        verticalLayout_20->addLayout(verticalLayout_13);


        verticalLayout_9->addWidget(widget);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        horizontalLayout_7->addWidget(scrollArea_2);

        Tabs->addTab(evolution_controls_tab, QString());
        statistics_tab = new QWidget();
        statistics_tab->setObjectName(QString::fromUtf8("statistics_tab"));
        Tabs->addTab(statistics_tab, QString());
        simulation_settings_tab = new QWidget();
        simulation_settings_tab->setObjectName(QString::fromUtf8("simulation_settings_tab"));
        verticalLayout_7 = new QVBoxLayout(simulation_settings_tab);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(simulation_settings_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setLineWidth(0);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 420, 192));
        QSizePolicy sizePolicy4(QSizePolicy::Ignored, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(scrollAreaWidgetContents->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents->setSizePolicy(sizePolicy4);
        verticalLayout_5 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        widget_2 = new QWidget(scrollAreaWidgetContents);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        verticalLayout_19 = new QVBoxLayout(widget_2);
        verticalLayout_19->setSpacing(0);
        verticalLayout_19->setObjectName(QString::fromUtf8("verticalLayout_19"));
        verticalLayout_19->setContentsMargins(9, 9, 9, 9);
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        rb_single_thread_mode = new QRadioButton(widget_2);
        simulation_modes = new QButtonGroup(MainWindow);
        simulation_modes->setObjectName(QString::fromUtf8("simulation_modes"));
        simulation_modes->addButton(rb_single_thread_mode);
        rb_single_thread_mode->setObjectName(QString::fromUtf8("rb_single_thread_mode"));

        horizontalLayout_2->addWidget(rb_single_thread_mode);

        rb_multi_thread_mode = new QRadioButton(widget_2);
        simulation_modes->addButton(rb_multi_thread_mode);
        rb_multi_thread_mode->setObjectName(QString::fromUtf8("rb_multi_thread_mode"));
        rb_multi_thread_mode->setChecked(true);

        horizontalLayout_2->addWidget(rb_multi_thread_mode);

        rb_cuda_mode = new QRadioButton(widget_2);
        simulation_modes->addButton(rb_cuda_mode);
        rb_cuda_mode->setObjectName(QString::fromUtf8("rb_cuda_mode"));

        horizontalLayout_2->addWidget(rb_cuda_mode);


        verticalLayout_19->addLayout(horizontalLayout_2);

        cb_stop_console_output = new QCheckBox(widget_2);
        cb_stop_console_output->setObjectName(QString::fromUtf8("cb_stop_console_output"));

        verticalLayout_19->addWidget(cb_stop_console_output);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label = new QLabel(widget_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_4->addWidget(label);

        le_num_threads = new QLineEdit(widget_2);
        le_num_threads->setObjectName(QString::fromUtf8("le_num_threads"));

        horizontalLayout_4->addWidget(le_num_threads);


        verticalLayout_19->addLayout(horizontalLayout_4);

        cb_synchronise_sim_and_win = new QCheckBox(widget_2);
        cb_synchronise_sim_and_win->setObjectName(QString::fromUtf8("cb_synchronise_sim_and_win"));
        cb_synchronise_sim_and_win->setChecked(false);

        verticalLayout_19->addWidget(cb_synchronise_sim_and_win);


        verticalLayout_5->addWidget(widget_2);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_7->addWidget(scrollArea);

        Tabs->addTab(simulation_settings_tab, QString());

        verticalLayout_4->addWidget(Tabs);


        horizontalLayout->addWidget(frame_3);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 3);
        frame_3->raise();
        frame_2->raise();

        verticalLayout_2->addWidget(horizontalFrame);

        verticalLayout_2->setStretch(0, 3);
        verticalLayout_2->setStretch(1, 1);
#ifndef QT_NO_SHORTCUT
        lb_fps->setBuddy(le_fps);
#endif // QT_NO_SHORTCUT

        retranslateUi(MainWindow);
        QObject::connect(b_reset, SIGNAL(clicked()), MainWindow, SLOT(b_reset_slot()));
        QObject::connect(rb_food, SIGNAL(clicked()), MainWindow, SLOT(rb_food_slot()));
        QObject::connect(cb_stop_console_output, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_stop_console_output_slot(bool)));
        QObject::connect(cb_synchronise_sim_and_win, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_synchronise_simulation_and_window_slot(bool)));
        QObject::connect(tb_stoprender, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_stoprender_slot(bool)));
        QObject::connect(rb_multi_thread_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_multi_thread_slot()));
        QObject::connect(rb_cuda_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_cuda_slot()));
        QObject::connect(rb_wall, SIGNAL(clicked()), MainWindow, SLOT(rb_wall_slot()));
        QObject::connect(b_reset_view, SIGNAL(clicked()), MainWindow, SLOT(b_reset_view_slot()));
        QObject::connect(rb_kill, SIGNAL(clicked()), MainWindow, SLOT(rb_kill_slot()));
        QObject::connect(le_fps, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_fps_slot()));
        QObject::connect(b_clear, SIGNAL(clicked()), MainWindow, SLOT(b_clear_slot()));
        QObject::connect(rb_single_thread_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_single_thread_slot()));
        QObject::connect(tb_pause, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_pause_slot(bool)));
        QObject::connect(le_num_threads, SIGNAL(returnPressed()), MainWindow, SLOT(le_num_threads_slot()));
        QObject::connect(le_sps, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_sps_slot()));
        QObject::connect(b_pass_one_tick, SIGNAL(clicked()), MainWindow, SLOT(b_pass_one_tick_slot()));
        QObject::connect(le_food_production_probability, SIGNAL(returnPressed()), MainWindow, SLOT(le_food_production_probability_slot()));
        QObject::connect(le_lifespan_multiplier, SIGNAL(returnPressed()), MainWindow, SLOT(le_lifespan_multiplier_slot()));
        QObject::connect(le_look_range, SIGNAL(returnPressed()), MainWindow, SLOT(le_look_range_slot()));
        QObject::connect(le_auto_food_drop_rate, SIGNAL(returnPressed()), MainWindow, SLOT(le_auto_food_drop_rate_slot()));
        QObject::connect(le_global_mutation_rate, SIGNAL(returnPressed()), MainWindow, SLOT(le_global_mutation_rate_slot()));
        QObject::connect(le_add, SIGNAL(returnPressed()), MainWindow, SLOT(le_add_cell_slot()));
        QObject::connect(le_change, SIGNAL(returnPressed()), MainWindow, SLOT(le_change_cell_slot()));
        QObject::connect(le_remove, SIGNAL(returnPressed()), MainWindow, SLOT(le_remove_cell_slot()));
        QObject::connect(cb_rotation_enabled, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_rotation_enabled_slot(bool)));
        QObject::connect(cb_on_touch_kill, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_on_touch_kill_slot(bool)));
        QObject::connect(cb_use_evoved_mutation_rate, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_use_evolved_mutation_rate_slot(bool)));
        QObject::connect(cb_movers_can_produce_food, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_movers_can_produce_food_slot(bool)));
        QObject::connect(cb_food_blocks_reproduction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_food_blocks_reproduction_slot(bool)));
        QObject::connect(le_extra_reproduction_cost, SIGNAL(returnPressed()), MainWindow, SLOT(le_extra_reproduction_cost_slot()));
        QObject::connect(le_cell_size, SIGNAL(returnPressed()), MainWindow, SLOT(le_cell_size_slot()));
        QObject::connect(le_simulation_width, SIGNAL(returnPressed()), MainWindow, SLOT(le_simulation_width_slot()));
        QObject::connect(le_simulation_height, SIGNAL(returnPressed()), MainWindow, SLOT(le_simulation_height_slot()));
        QObject::connect(cb_fill_window, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_fill_window_slot(bool)));
        QObject::connect(b_resize_and_reset, SIGNAL(clicked()), MainWindow, SLOT(b_resize_and_reset_slot()));
        QObject::connect(cb_reset_on_total_extinction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_reset_on_total_extinction_slot(bool)));
        QObject::connect(cb_pause_on_total_extinction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_pause_on_total_extinction_slot(bool)));
        QObject::connect(b_generate_random_walls, SIGNAL(clicked()), MainWindow, SLOT(b_generate_random_walls_slot()));
        QObject::connect(b_clear_all_walls, SIGNAL(clicked()), MainWindow, SLOT(b_clear_all_walls_slot()));
        QObject::connect(cb_clear_walls_on_reset, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_clear_walls_on_reset_slot(bool)));
        QObject::connect(b_save_world, SIGNAL(clicked()), MainWindow, SLOT(b_save_world_slot()));
        QObject::connect(b_load_world, SIGNAL(clicked()), MainWindow, SLOT(b_load_world_slot()));
        QObject::connect(cb_override_evolution_controls, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_override_evolution_controls_slot(bool)));
        QObject::connect(cb_generate_random_walls_on_reset, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_generate_random_walls_on_reset_slot(bool)));

        Tabs->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        lb_fps->setText(QApplication::translate("MainWindow", "fps:", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "Set max fps:", nullptr));
        le_fps->setText(QApplication::translate("MainWindow", "60", nullptr));
        lb_sps->setText(QApplication::translate("MainWindow", "sps:", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "Set max sps:", nullptr));
        le_sps->setText(QApplication::translate("MainWindow", "-1", nullptr));
        b_clear->setText(QApplication::translate("MainWindow", "Clear", nullptr));
        b_reset->setText(QApplication::translate("MainWindow", "Reset", nullptr));
        tb_pause->setText(QApplication::translate("MainWindow", "Pause", nullptr));
        tb_stoprender->setText(QApplication::translate("MainWindow", "Stop render", nullptr));
        b_reset_view->setText(QApplication::translate("MainWindow", "Reset view", nullptr));
        b_pass_one_tick->setText(QApplication::translate("MainWindow", "Pass one tick", nullptr));
        rb_food->setText(QApplication::translate("MainWindow", "Food mode", nullptr));
        rb_kill->setText(QApplication::translate("MainWindow", "Kill mode", nullptr));
        rb_wall->setText(QApplication::translate("MainWindow", "Wall mode", nullptr));
        textEdit->setHtml(QApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This is c++ implementation of an original The Life Engine (https://thelifeengine.net/) made in javascript (https://github.com/MaxRobinsonTheGreat/LifeEngine).</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This version will (in the future) feature everything that the original version has, but I (maybe) will also add some new features (in the far future).</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent"
                        ":0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Important Information:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">sps is the number of simulation ticks per second.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">setting max sps/sps to &lt;0 will enable unlimited mode.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Not implemented:</p></body></html>", nullptr));
        Tabs->setTabText(Tabs->indexOf(about_tab), QApplication::translate("MainWindow", "About", nullptr));
        Tabs->setTabText(Tabs->indexOf(editor_tab), QApplication::translate("MainWindow", "Editor", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "Cell size:", nullptr));
        le_cell_size->setText(QApplication::translate("MainWindow", "1", nullptr));
        cb_fill_window->setText(QApplication::translate("MainWindow", "Fill window", nullptr));
        label_13->setText(QApplication::translate("MainWindow", "Width:", nullptr));
        le_simulation_width->setText(QApplication::translate("MainWindow", "600", nullptr));
        label_14->setText(QApplication::translate("MainWindow", "Height:", nullptr));
        le_simulation_height->setText(QApplication::translate("MainWindow", "600", nullptr));
        b_resize_and_reset->setText(QApplication::translate("MainWindow", "Resize and Reset", nullptr));
        cb_reset_on_total_extinction->setText(QApplication::translate("MainWindow", "Reset on total extinction", nullptr));
        lb_auto_reset_count->setText(QApplication::translate("MainWindow", "Auto reset count: 0", nullptr));
        cb_pause_on_total_extinction->setText(QApplication::translate("MainWindow", "Pause on total extinction", nullptr));
        b_generate_random_walls->setText(QApplication::translate("MainWindow", "Generate random walls", nullptr));
        cb_generate_random_walls_on_reset->setText(QApplication::translate("MainWindow", "Generate random walls on reset", nullptr));
        b_clear_all_walls->setText(QApplication::translate("MainWindow", "Clear all walls", nullptr));
        cb_clear_walls_on_reset->setText(QApplication::translate("MainWindow", "Clear walls on reset", nullptr));
        b_save_world->setText(QApplication::translate("MainWindow", "Save world", nullptr));
        b_load_world->setText(QApplication::translate("MainWindow", "Load world", nullptr));
        cb_override_evolution_controls->setText(QApplication::translate("MainWindow", "Override Evolution Controls", nullptr));
        Tabs->setTabText(Tabs->indexOf(world_controls_tab), QApplication::translate("MainWindow", "World Controls", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "food production probability:", nullptr));
        le_food_production_probability->setText(QApplication::translate("MainWindow", "5", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "llifespan multiplier:", nullptr));
        le_lifespan_multiplier->setText(QApplication::translate("MainWindow", "100", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "look range:", nullptr));
        le_look_range->setText(QApplication::translate("MainWindow", "50", nullptr));
        label_9->setText(QApplication::translate("MainWindow", "auto food drop rate:", nullptr));
        le_auto_food_drop_rate->setText(QApplication::translate("MainWindow", "0", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "extra reproduction cost:", nullptr));
        le_extra_reproduction_cost->setText(QApplication::translate("MainWindow", "0", nullptr));
        cb_use_evoved_mutation_rate->setText(QApplication::translate("MainWindow", "Use evolved mutation rate", nullptr));
        lb_mutation_rate->setText(QApplication::translate("MainWindow", "global mutation rate:", nullptr));
        le_global_mutation_rate->setText(QApplication::translate("MainWindow", "5", nullptr));
        label_10->setText(QApplication::translate("MainWindow", "add cell:", nullptr));
        le_add->setText(QApplication::translate("MainWindow", "33", nullptr));
        label_11->setText(QApplication::translate("MainWindow", "change cell:", nullptr));
        le_change->setText(QApplication::translate("MainWindow", "33", nullptr));
        label_12->setText(QApplication::translate("MainWindow", "remove cell:", nullptr));
        le_remove->setText(QApplication::translate("MainWindow", "33", nullptr));
        cb_rotation_enabled->setText(QApplication::translate("MainWindow", "Rotation enabled", nullptr));
        cb_on_touch_kill->setText(QApplication::translate("MainWindow", "On touch kill", nullptr));
        cb_movers_can_produce_food->setText(QApplication::translate("MainWindow", "Movers can produce food", nullptr));
        cb_food_blocks_reproduction->setText(QApplication::translate("MainWindow", "Food blocks reproduction", nullptr));
        Tabs->setTabText(Tabs->indexOf(evolution_controls_tab), QApplication::translate("MainWindow", "Evolution Controls", nullptr));
        Tabs->setTabText(Tabs->indexOf(statistics_tab), QApplication::translate("MainWindow", "Statistics", nullptr));
        rb_single_thread_mode->setText(QApplication::translate("MainWindow", "Single thread CPU", nullptr));
        rb_multi_thread_mode->setText(QApplication::translate("MainWindow", "Multi-thread CPU", nullptr));
        rb_cuda_mode->setText(QApplication::translate("MainWindow", "CUDA", nullptr));
        cb_stop_console_output->setText(QApplication::translate("MainWindow", "Stop console output", nullptr));
        label->setText(QApplication::translate("MainWindow", "Set number of CPU threads:", nullptr));
        le_num_threads->setText(QApplication::translate("MainWindow", "1", nullptr));
        cb_synchronise_sim_and_win->setText(QApplication::translate("MainWindow", "Synchronise simulation and window", nullptr));
        Tabs->setTabText(Tabs->indexOf(simulation_settings_tab), QApplication::translate("MainWindow", "Simulation Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WINDOWUI_H
