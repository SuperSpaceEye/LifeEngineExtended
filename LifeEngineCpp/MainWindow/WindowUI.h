/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef WINDOWUI_H
#define WINDOWUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QVBoxLayout *verticalLayout_2;
    QGraphicsView *simulation_graphicsView;
    QFrame *menu_frame;
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
    QPushButton *b_reset;
    QPushButton *b_reset_view;
    QPushButton *b_pass_one_tick;
    QPushButton *tb_stoprender;
    QPushButton *b_kill_all_organisms_slot;
    QPushButton *tb_pause;
    QPushButton *b_clear;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *rb_food;
    QRadioButton *rb_kill;
    QRadioButton *rb_wall;
    QRadioButton *rb_null_button;
    QHBoxLayout *horizontalLayout_41;
    QLabel *label_43;
    QLineEdit *le_brush_size;
    QCheckBox *cb_synchronise_sim_and_win;
    QFrame *frame_3;
    QVBoxLayout *verticalLayout_4;
    QTabWidget *Tabs;
    QWidget *about_tab;
    QVBoxLayout *verticalLayout_8;
    QTextEdit *textEdit;
    QWidget *editor_tab;
    QVBoxLayout *verticalLayout_10;
    QPushButton *tb_open_organism_editor;
    QCheckBox *cb_editor_always_on_top;
    QWidget *world_controls_tab;
    QVBoxLayout *verticalLayout;
    QScrollArea *scrollArea_3;
    QWidget *scrollAreaWidgetContents_3;
    QVBoxLayout *verticalLayout_12;
    QWidget *widget_3;
    QVBoxLayout *verticalLayout_31;
    QHBoxLayout *horizontalLayout_45;
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
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_18;
    QLineEdit *le_max_organisms;
    QVBoxLayout *verticalLayout_22;
    QPushButton *b_generate_random_walls;
    QCheckBox *cb_generate_random_walls_on_reset;
    QPushButton *b_clear_all_walls;
    QCheckBox *cb_clear_walls_on_reset;
    QPushButton *b_save_world;
    QHBoxLayout *horizontalLayout_21;
    QPushButton *b_load_world;
    QSpacerItem *verticalSpacer;
    QLabel *label_45;
    QTableWidget *table_organism_block_parameters;
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
    QHBoxLayout *horizontalLayout_24;
    QLabel *label_15;
    QLineEdit *le_produce_food_every_n_tick;
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
    QLineEdit *le_auto_produce_n_food;
    QHBoxLayout *horizontalLayout_42;
    QLabel *label_44;
    QLineEdit *le_auto_produce_food_every_n_tick;
    QHBoxLayout *horizontalLayout_18;
    QLabel *label_6;
    QLineEdit *le_extra_reproduction_cost;
    QCheckBox *cb_use_evolved_anatomy_mutation_rate;
    QVBoxLayout *mutation_rate_layout;
    QHBoxLayout *horizontalLayout_12;
    QLabel *lb_mutation_rate;
    QLineEdit *le_global_anatomy_mutation_rate;
    QHBoxLayout *horizontalLayout_28;
    QLabel *label_23;
    QLineEdit *le_anatomy_mutation_rate_delimiter;
    QCheckBox *cb_use_evolved_brain_mutation_rate;
    QHBoxLayout *horizontalLayout_27;
    QLabel *label_20;
    QLineEdit *le_global_brain_mutation_rate;
    QHBoxLayout *horizontalLayout_29;
    QLabel *label_24;
    QLineEdit *le_brain_mutation_rate_delimiter;
    QHBoxLayout *horizontalLayout_19;
    QLabel *label_21;
    QLineEdit *le_killer_damage_amount;
    QVBoxLayout *verticalLayout_17;
    QHBoxLayout *horizontalLayout_16;
    QLabel *label_10;
    QLineEdit *le_add;
    QLabel *label_11;
    QLineEdit *le_change;
    QLabel *label_12;
    QLineEdit *le_remove;
    QCheckBox *cb_runtime_rotation_enabled;
    QCheckBox *cb_reproducing_rotation_enabled;
    QCheckBox *cb_on_touch_kill;
    QCheckBox *cb_movers_can_produce_food;
    QCheckBox *cb_food_blocks_reproduction;
    QCheckBox *cb_food_blocks_movement;
    QVBoxLayout *verticalLayout_26;
    QHBoxLayout *horizontalLayout_25;
    QLabel *label_17;
    QLineEdit *le_min_reproduction_distance;
    QHBoxLayout *horizontalLayout_26;
    QLabel *label_16;
    QLineEdit *le_max_reproduction_distance;
    QCheckBox *cb_fix_reproduction_distance;
    QCheckBox *cb_self_organism_blocks_block_sight;
    QVBoxLayout *verticalLayout_34;
    QCheckBox *cb_set_fixed_move_range;
    QHBoxLayout *horizontalLayout_39;
    QLabel *label_41;
    QLineEdit *le_min_move_range;
    QHBoxLayout *horizontalLayout_40;
    QLabel *label_40;
    QLineEdit *le_max_move_range;
    QHBoxLayout *horizontalLayout_38;
    QLabel *label_42;
    QLineEdit *le_move_range_delimiter;
    QCheckBox *cb_failed_reproduction_eats_food;
    QCheckBox *cb_rotate_every_move_tick;
    QCheckBox *cb_multiply_food_production_prob;
    QCheckBox *cb_simplified_food_production;
    QCheckBox *cb_stop_when_one_food_generated;
    QCheckBox *cb_eat_then_produce;
    QWidget *statistics_tab;
    QVBoxLayout *verticalLayout_3;
    QScrollArea *scrollArea_4;
    QWidget *scrollAreaWidgetContents_4;
    QVBoxLayout *verticalLayout_24;
    QWidget *widget_5;
    QVBoxLayout *verticalLayout_37;
    QPushButton *tb_open_statisctics;
    QCheckBox *cb_statistics_always_on_top;
    QWidget *settings_tab;
    QVBoxLayout *verticalLayout_7;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_5;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_19;
    QVBoxLayout *verticalLayout_23;
    QRadioButton *rb_single_thread_mode;
    QRadioButton *rb_partial_multi_thread_mode;
    QRadioButton *rb_multi_thread_mode;
    QRadioButton *rb_cuda_mode;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label;
    QLineEdit *le_num_threads;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_19;
    QLineEdit *le_float_number_precision;
    QCheckBox *cb_disable_warnings;
    QCheckBox *cb_wait_for_engine_to_stop;
    QHBoxLayout *horizontalLayout_30;
    QLabel *label_25;
    QLineEdit *le_font_size;
    QCheckBox *cb_simplified_rendering;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_46;
    QLineEdit *le_update_info_every_n_milliseconds;
    QCheckBox *cb_synchronise_info_with_window;
    QCheckBox *cb_use_nvidia_for_image_generation;
    QHBoxLayout *horizontalLayout_14;
    QLabel *label_47;
    QLineEdit *le_menu_height;
    QSpacerItem *verticalSpacer_2;
    QLabel *label_22;
    QHBoxLayout *horizontalLayout_20;
    QLabel *label_26;
    QLineEdit *le_perlin_octaves;
    QHBoxLayout *horizontalLayout_31;
    QLabel *label_27;
    QLineEdit *le_perlin_persistence;
    QHBoxLayout *horizontalLayout_32;
    QLabel *label_28;
    QLineEdit *le_perlin_upper_bound;
    QHBoxLayout *horizontalLayout_33;
    QLabel *label_29;
    QLineEdit *le_perlin_lower_bound;
    QHBoxLayout *horizontalLayout_34;
    QLabel *label_30;
    QLineEdit *le_perlin_x_modifier;
    QHBoxLayout *horizontalLayout_17;
    QLabel *label_31;
    QLineEdit *le_perlin_y_modifier;
    QButtonGroup *cursor_modes;
    QButtonGroup *simulation_modes;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(898, 900);
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(0, 0));
        verticalLayout_2 = new QVBoxLayout(MainWindow);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setSizeConstraint(QLayout::SetMaximumSize);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        simulation_graphicsView = new QGraphicsView(MainWindow);
        simulation_graphicsView->setObjectName(QStringLiteral("simulation_graphicsView"));
        sizePolicy.setHeightForWidth(simulation_graphicsView->sizePolicy().hasHeightForWidth());
        simulation_graphicsView->setSizePolicy(sizePolicy);
        simulation_graphicsView->setLayoutDirection(Qt::LeftToRight);
        simulation_graphicsView->setStyleSheet(QStringLiteral(""));
        simulation_graphicsView->setFrameShape(QFrame::NoFrame);
        simulation_graphicsView->setFrameShadow(QFrame::Plain);
        simulation_graphicsView->setLineWidth(0);
        simulation_graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        simulation_graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        simulation_graphicsView->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
        simulation_graphicsView->setInteractive(false);
        simulation_graphicsView->setOptimizationFlags(QGraphicsView::DontAdjustForAntialiasing);

        verticalLayout_2->addWidget(simulation_graphicsView);

        menu_frame = new QFrame(MainWindow);
        menu_frame->setObjectName(QStringLiteral("menu_frame"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(menu_frame->sizePolicy().hasHeightForWidth());
        menu_frame->setSizePolicy(sizePolicy1);
        menu_frame->setMinimumSize(QSize(0, 250));
        menu_frame->setFrameShape(QFrame::NoFrame);
        horizontalLayout = new QHBoxLayout(menu_frame);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        frame_2 = new QFrame(menu_frame);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(frame_2->sizePolicy().hasHeightForWidth());
        frame_2->setSizePolicy(sizePolicy2);
        frame_2->setMinimumSize(QSize(0, 0));
        frame_2->setFrameShape(QFrame::NoFrame);
        frame_2->setFrameShadow(QFrame::Sunken);
        frame_2->setLineWidth(0);
        verticalLayout_6 = new QVBoxLayout(frame_2);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(9, 9, 9, 9);
        lb_fps = new QLabel(frame_2);
        lb_fps->setObjectName(QStringLiteral("lb_fps"));

        verticalLayout_6->addWidget(lb_fps);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        label_2 = new QLabel(frame_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout_6->addWidget(label_2);

        le_fps = new QLineEdit(frame_2);
        le_fps->setObjectName(QStringLiteral("le_fps"));

        horizontalLayout_6->addWidget(le_fps);


        verticalLayout_6->addLayout(horizontalLayout_6);

        lb_sps = new QLabel(frame_2);
        lb_sps->setObjectName(QStringLiteral("lb_sps"));

        verticalLayout_6->addWidget(lb_sps);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        label_3 = new QLabel(frame_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        horizontalLayout_5->addWidget(label_3);

        le_sps = new QLineEdit(frame_2);
        le_sps->setObjectName(QStringLiteral("le_sps"));
        le_sps->setEnabled(true);

        horizontalLayout_5->addWidget(le_sps);


        verticalLayout_6->addLayout(horizontalLayout_5);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        b_reset = new QPushButton(frame_2);
        b_reset->setObjectName(QStringLiteral("b_reset"));

        gridLayout->addWidget(b_reset, 0, 3, 1, 1);

        b_reset_view = new QPushButton(frame_2);
        b_reset_view->setObjectName(QStringLiteral("b_reset_view"));

        gridLayout->addWidget(b_reset_view, 1, 0, 1, 1);

        b_pass_one_tick = new QPushButton(frame_2);
        b_pass_one_tick->setObjectName(QStringLiteral("b_pass_one_tick"));

        gridLayout->addWidget(b_pass_one_tick, 1, 1, 1, 1);

        tb_stoprender = new QPushButton(frame_2);
        tb_stoprender->setObjectName(QStringLiteral("tb_stoprender"));
        tb_stoprender->setCheckable(true);

        gridLayout->addWidget(tb_stoprender, 0, 1, 1, 1);

        b_kill_all_organisms_slot = new QPushButton(frame_2);
        b_kill_all_organisms_slot->setObjectName(QStringLiteral("b_kill_all_organisms_slot"));

        gridLayout->addWidget(b_kill_all_organisms_slot, 1, 2, 1, 1);

        tb_pause = new QPushButton(frame_2);
        tb_pause->setObjectName(QStringLiteral("tb_pause"));
        tb_pause->setCheckable(true);

        gridLayout->addWidget(tb_pause, 0, 0, 1, 1);

        b_clear = new QPushButton(frame_2);
        b_clear->setObjectName(QStringLiteral("b_clear"));

        gridLayout->addWidget(b_clear, 0, 2, 1, 1);


        verticalLayout_6->addLayout(gridLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        rb_food = new QRadioButton(frame_2);
        cursor_modes = new QButtonGroup(MainWindow);
        cursor_modes->setObjectName(QStringLiteral("cursor_modes"));
        cursor_modes->setExclusive(true);
        cursor_modes->addButton(rb_food);
        rb_food->setObjectName(QStringLiteral("rb_food"));
        rb_food->setEnabled(true);
        rb_food->setChecked(true);

        horizontalLayout_3->addWidget(rb_food);

        rb_kill = new QRadioButton(frame_2);
        cursor_modes->addButton(rb_kill);
        rb_kill->setObjectName(QStringLiteral("rb_kill"));
        rb_kill->setEnabled(true);

        horizontalLayout_3->addWidget(rb_kill);

        rb_wall = new QRadioButton(frame_2);
        cursor_modes->addButton(rb_wall);
        rb_wall->setObjectName(QStringLiteral("rb_wall"));
        rb_wall->setEnabled(true);

        horizontalLayout_3->addWidget(rb_wall);

        rb_null_button = new QRadioButton(frame_2);
        cursor_modes->addButton(rb_null_button);
        rb_null_button->setObjectName(QStringLiteral("rb_null_button"));
        rb_null_button->setEnabled(false);
        rb_null_button->setMaximumSize(QSize(0, 0));

        horizontalLayout_3->addWidget(rb_null_button);


        verticalLayout_6->addLayout(horizontalLayout_3);

        horizontalLayout_41 = new QHBoxLayout();
        horizontalLayout_41->setObjectName(QStringLiteral("horizontalLayout_41"));
        label_43 = new QLabel(frame_2);
        label_43->setObjectName(QStringLiteral("label_43"));

        horizontalLayout_41->addWidget(label_43);

        le_brush_size = new QLineEdit(frame_2);
        le_brush_size->setObjectName(QStringLiteral("le_brush_size"));

        horizontalLayout_41->addWidget(le_brush_size);


        verticalLayout_6->addLayout(horizontalLayout_41);

        cb_synchronise_sim_and_win = new QCheckBox(frame_2);
        cb_synchronise_sim_and_win->setObjectName(QStringLiteral("cb_synchronise_sim_and_win"));
        cb_synchronise_sim_and_win->setChecked(false);

        verticalLayout_6->addWidget(cb_synchronise_sim_and_win);


        horizontalLayout->addWidget(frame_2);

        frame_3 = new QFrame(menu_frame);
        frame_3->setObjectName(QStringLiteral("frame_3"));
        frame_3->setFrameShape(QFrame::StyledPanel);
        frame_3->setFrameShadow(QFrame::Raised);
        verticalLayout_4 = new QVBoxLayout(frame_3);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        Tabs = new QTabWidget(frame_3);
        Tabs->setObjectName(QStringLiteral("Tabs"));
        Tabs->setTabPosition(QTabWidget::North);
        Tabs->setTabShape(QTabWidget::Rounded);
        Tabs->setIconSize(QSize(16, 16));
        Tabs->setElideMode(Qt::ElideNone);
        Tabs->setDocumentMode(true);
        Tabs->setTabsClosable(false);
        Tabs->setMovable(false);
        Tabs->setTabBarAutoHide(true);
        about_tab = new QWidget();
        about_tab->setObjectName(QStringLiteral("about_tab"));
        verticalLayout_8 = new QVBoxLayout(about_tab);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        textEdit = new QTextEdit(about_tab);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        textEdit->setReadOnly(true);
        textEdit->setOverwriteMode(false);

        verticalLayout_8->addWidget(textEdit);

        Tabs->addTab(about_tab, QString());
        editor_tab = new QWidget();
        editor_tab->setObjectName(QStringLiteral("editor_tab"));
        editor_tab->setLayoutDirection(Qt::LeftToRight);
        verticalLayout_10 = new QVBoxLayout(editor_tab);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        tb_open_organism_editor = new QPushButton(editor_tab);
        tb_open_organism_editor->setObjectName(QStringLiteral("tb_open_organism_editor"));
        tb_open_organism_editor->setCheckable(true);

        verticalLayout_10->addWidget(tb_open_organism_editor);

        cb_editor_always_on_top = new QCheckBox(editor_tab);
        cb_editor_always_on_top->setObjectName(QStringLiteral("cb_editor_always_on_top"));

        verticalLayout_10->addWidget(cb_editor_always_on_top);

        Tabs->addTab(editor_tab, QString());
        world_controls_tab = new QWidget();
        world_controls_tab->setObjectName(QStringLiteral("world_controls_tab"));
        verticalLayout = new QVBoxLayout(world_controls_tab);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea_3 = new QScrollArea(world_controls_tab);
        scrollArea_3->setObjectName(QStringLiteral("scrollArea_3"));
        scrollArea_3->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName(QStringLiteral("scrollAreaWidgetContents_3"));
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 504, 452));
        QSizePolicy sizePolicy3(QSizePolicy::Ignored, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(scrollAreaWidgetContents_3->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_3->setSizePolicy(sizePolicy3);
        verticalLayout_12 = new QVBoxLayout(scrollAreaWidgetContents_3);
        verticalLayout_12->setSpacing(0);
        verticalLayout_12->setObjectName(QStringLiteral("verticalLayout_12"));
        verticalLayout_12->setContentsMargins(0, 0, 0, 0);
        widget_3 = new QWidget(scrollAreaWidgetContents_3);
        widget_3->setObjectName(QStringLiteral("widget_3"));
        QSizePolicy sizePolicy4(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(widget_3->sizePolicy().hasHeightForWidth());
        widget_3->setSizePolicy(sizePolicy4);
        widget_3->setMinimumSize(QSize(0, 0));
        verticalLayout_31 = new QVBoxLayout(widget_3);
        verticalLayout_31->setObjectName(QStringLiteral("verticalLayout_31"));
        verticalLayout_31->setContentsMargins(6, 6, 6, 0);
        horizontalLayout_45 = new QHBoxLayout();
        horizontalLayout_45->setObjectName(QStringLiteral("horizontalLayout_45"));
        verticalLayout_21 = new QVBoxLayout();
        verticalLayout_21->setObjectName(QStringLiteral("verticalLayout_21"));
        horizontalLayout_22 = new QHBoxLayout();
        horizontalLayout_22->setObjectName(QStringLiteral("horizontalLayout_22"));
        label_8 = new QLabel(widget_3);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_22->addWidget(label_8);

        le_cell_size = new QLineEdit(widget_3);
        le_cell_size->setObjectName(QStringLiteral("le_cell_size"));
        le_cell_size->setEnabled(true);

        horizontalLayout_22->addWidget(le_cell_size);

        cb_fill_window = new QCheckBox(widget_3);
        cb_fill_window->setObjectName(QStringLiteral("cb_fill_window"));
        cb_fill_window->setEnabled(true);

        horizontalLayout_22->addWidget(cb_fill_window);


        verticalLayout_21->addLayout(horizontalLayout_22);

        horizontalLayout_23 = new QHBoxLayout();
        horizontalLayout_23->setObjectName(QStringLiteral("horizontalLayout_23"));
        label_13 = new QLabel(widget_3);
        label_13->setObjectName(QStringLiteral("label_13"));

        horizontalLayout_23->addWidget(label_13);

        le_simulation_width = new QLineEdit(widget_3);
        le_simulation_width->setObjectName(QStringLiteral("le_simulation_width"));

        horizontalLayout_23->addWidget(le_simulation_width);

        label_14 = new QLabel(widget_3);
        label_14->setObjectName(QStringLiteral("label_14"));

        horizontalLayout_23->addWidget(label_14);

        le_simulation_height = new QLineEdit(widget_3);
        le_simulation_height->setObjectName(QStringLiteral("le_simulation_height"));

        horizontalLayout_23->addWidget(le_simulation_height);


        verticalLayout_21->addLayout(horizontalLayout_23);

        b_resize_and_reset = new QPushButton(widget_3);
        b_resize_and_reset->setObjectName(QStringLiteral("b_resize_and_reset"));

        verticalLayout_21->addWidget(b_resize_and_reset);

        cb_reset_on_total_extinction = new QCheckBox(widget_3);
        cb_reset_on_total_extinction->setObjectName(QStringLiteral("cb_reset_on_total_extinction"));

        verticalLayout_21->addWidget(cb_reset_on_total_extinction);

        lb_auto_reset_count = new QLabel(widget_3);
        lb_auto_reset_count->setObjectName(QStringLiteral("lb_auto_reset_count"));

        verticalLayout_21->addWidget(lb_auto_reset_count);

        cb_pause_on_total_extinction = new QCheckBox(widget_3);
        cb_pause_on_total_extinction->setObjectName(QStringLiteral("cb_pause_on_total_extinction"));

        verticalLayout_21->addWidget(cb_pause_on_total_extinction);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        label_18 = new QLabel(widget_3);
        label_18->setObjectName(QStringLiteral("label_18"));

        horizontalLayout_11->addWidget(label_18);

        le_max_organisms = new QLineEdit(widget_3);
        le_max_organisms->setObjectName(QStringLiteral("le_max_organisms"));

        horizontalLayout_11->addWidget(le_max_organisms);


        verticalLayout_21->addLayout(horizontalLayout_11);


        horizontalLayout_45->addLayout(verticalLayout_21);

        verticalLayout_22 = new QVBoxLayout();
        verticalLayout_22->setObjectName(QStringLiteral("verticalLayout_22"));
        verticalLayout_22->setSizeConstraint(QLayout::SetMinimumSize);
        verticalLayout_22->setContentsMargins(-1, -1, -1, 0);
        b_generate_random_walls = new QPushButton(widget_3);
        b_generate_random_walls->setObjectName(QStringLiteral("b_generate_random_walls"));
        b_generate_random_walls->setEnabled(true);

        verticalLayout_22->addWidget(b_generate_random_walls);

        cb_generate_random_walls_on_reset = new QCheckBox(widget_3);
        cb_generate_random_walls_on_reset->setObjectName(QStringLiteral("cb_generate_random_walls_on_reset"));
        cb_generate_random_walls_on_reset->setEnabled(true);

        verticalLayout_22->addWidget(cb_generate_random_walls_on_reset);

        b_clear_all_walls = new QPushButton(widget_3);
        b_clear_all_walls->setObjectName(QStringLiteral("b_clear_all_walls"));
        b_clear_all_walls->setEnabled(true);

        verticalLayout_22->addWidget(b_clear_all_walls);

        cb_clear_walls_on_reset = new QCheckBox(widget_3);
        cb_clear_walls_on_reset->setObjectName(QStringLiteral("cb_clear_walls_on_reset"));
        cb_clear_walls_on_reset->setEnabled(true);

        verticalLayout_22->addWidget(cb_clear_walls_on_reset);

        b_save_world = new QPushButton(widget_3);
        b_save_world->setObjectName(QStringLiteral("b_save_world"));
        b_save_world->setEnabled(true);

        verticalLayout_22->addWidget(b_save_world);

        horizontalLayout_21 = new QHBoxLayout();
        horizontalLayout_21->setObjectName(QStringLiteral("horizontalLayout_21"));
        horizontalLayout_21->setSizeConstraint(QLayout::SetMinimumSize);
        b_load_world = new QPushButton(widget_3);
        b_load_world->setObjectName(QStringLiteral("b_load_world"));
        b_load_world->setEnabled(true);

        horizontalLayout_21->addWidget(b_load_world);


        verticalLayout_22->addLayout(horizontalLayout_21);


        horizontalLayout_45->addLayout(verticalLayout_22);

        horizontalLayout_45->setStretch(0, 1);
        horizontalLayout_45->setStretch(1, 1);

        verticalLayout_31->addLayout(horizontalLayout_45);

        verticalSpacer = new QSpacerItem(20, 10, QSizePolicy::Minimum, QSizePolicy::Minimum);

        verticalLayout_31->addItem(verticalSpacer);

        label_45 = new QLabel(widget_3);
        label_45->setObjectName(QStringLiteral("label_45"));
        label_45->setTextFormat(Qt::AutoText);
        label_45->setScaledContents(true);
        label_45->setAlignment(Qt::AlignCenter);
        label_45->setWordWrap(false);
        label_45->setOpenExternalLinks(false);

        verticalLayout_31->addWidget(label_45);

        table_organism_block_parameters = new QTableWidget(widget_3);
        if (table_organism_block_parameters->columnCount() < 3)
            table_organism_block_parameters->setColumnCount(3);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        table_organism_block_parameters->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        table_organism_block_parameters->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        table_organism_block_parameters->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        if (table_organism_block_parameters->rowCount() < 6)
            table_organism_block_parameters->setRowCount(6);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(0, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(1, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(2, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(3, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(4, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        table_organism_block_parameters->setVerticalHeaderItem(5, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        __qtablewidgetitem9->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEditable|Qt::ItemIsDragEnabled|Qt::ItemIsDropEnabled|Qt::ItemIsUserCheckable|Qt::ItemIsEnabled|Qt::ItemIsTristate);
        table_organism_block_parameters->setItem(0, 0, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(0, 1, __qtablewidgetitem10);
        QTableWidgetItem *__qtablewidgetitem11 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(0, 2, __qtablewidgetitem11);
        QTableWidgetItem *__qtablewidgetitem12 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(1, 0, __qtablewidgetitem12);
        QTableWidgetItem *__qtablewidgetitem13 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(1, 1, __qtablewidgetitem13);
        QTableWidgetItem *__qtablewidgetitem14 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(1, 2, __qtablewidgetitem14);
        QTableWidgetItem *__qtablewidgetitem15 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(2, 0, __qtablewidgetitem15);
        QTableWidgetItem *__qtablewidgetitem16 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(2, 1, __qtablewidgetitem16);
        QTableWidgetItem *__qtablewidgetitem17 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(2, 2, __qtablewidgetitem17);
        QTableWidgetItem *__qtablewidgetitem18 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(3, 0, __qtablewidgetitem18);
        QTableWidgetItem *__qtablewidgetitem19 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(3, 1, __qtablewidgetitem19);
        QTableWidgetItem *__qtablewidgetitem20 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(3, 2, __qtablewidgetitem20);
        QTableWidgetItem *__qtablewidgetitem21 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(4, 0, __qtablewidgetitem21);
        QTableWidgetItem *__qtablewidgetitem22 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(4, 1, __qtablewidgetitem22);
        QTableWidgetItem *__qtablewidgetitem23 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(4, 2, __qtablewidgetitem23);
        QTableWidgetItem *__qtablewidgetitem24 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(5, 0, __qtablewidgetitem24);
        QTableWidgetItem *__qtablewidgetitem25 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(5, 1, __qtablewidgetitem25);
        QTableWidgetItem *__qtablewidgetitem26 = new QTableWidgetItem();
        table_organism_block_parameters->setItem(5, 2, __qtablewidgetitem26);
        table_organism_block_parameters->setObjectName(QStringLiteral("table_organism_block_parameters"));
        table_organism_block_parameters->setAutoScroll(false);
        table_organism_block_parameters->setEditTriggers(QAbstractItemView::CurrentChanged);
        table_organism_block_parameters->setSelectionMode(QAbstractItemView::SingleSelection);
        table_organism_block_parameters->setShowGrid(true);
        table_organism_block_parameters->setGridStyle(Qt::SolidLine);
        table_organism_block_parameters->setWordWrap(false);
        table_organism_block_parameters->horizontalHeader()->setVisible(false);
        table_organism_block_parameters->horizontalHeader()->setCascadingSectionResizes(false);
        table_organism_block_parameters->horizontalHeader()->setDefaultSectionSize(120);
        table_organism_block_parameters->horizontalHeader()->setMinimumSectionSize(57);
        table_organism_block_parameters->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));
        table_organism_block_parameters->horizontalHeader()->setStretchLastSection(false);
        table_organism_block_parameters->verticalHeader()->setVisible(false);
        table_organism_block_parameters->verticalHeader()->setCascadingSectionResizes(false);
        table_organism_block_parameters->verticalHeader()->setDefaultSectionSize(21);
        table_organism_block_parameters->verticalHeader()->setHighlightSections(true);
        table_organism_block_parameters->verticalHeader()->setMinimumSectionSize(21);
        table_organism_block_parameters->verticalHeader()->setProperty("showSortIndicator", QVariant(false));
        table_organism_block_parameters->verticalHeader()->setStretchLastSection(false);

        verticalLayout_31->addWidget(table_organism_block_parameters);

        verticalLayout_31->setStretch(0, 1);

        verticalLayout_12->addWidget(widget_3);

        scrollArea_3->setWidget(scrollAreaWidgetContents_3);

        verticalLayout->addWidget(scrollArea_3);

        Tabs->addTab(world_controls_tab, QString());
        evolution_controls_tab = new QWidget();
        evolution_controls_tab->setObjectName(QStringLiteral("evolution_controls_tab"));
        horizontalLayout_7 = new QHBoxLayout(evolution_controls_tab);
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        scrollArea_2 = new QScrollArea(evolution_controls_tab);
        scrollArea_2->setObjectName(QStringLiteral("scrollArea_2"));
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
        scrollAreaWidgetContents_2->setObjectName(QStringLiteral("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, -585, 504, 1121));
        QSizePolicy sizePolicy5(QSizePolicy::Ignored, QSizePolicy::Expanding);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(scrollAreaWidgetContents_2->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_2->setSizePolicy(sizePolicy5);
        verticalLayout_9 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(9, 9, 9, 9);
        widget = new QWidget(scrollAreaWidgetContents_2);
        widget->setObjectName(QStringLiteral("widget"));
        QSizePolicy sizePolicy6(QSizePolicy::Maximum, QSizePolicy::Expanding);
        sizePolicy6.setHorizontalStretch(10);
        sizePolicy6.setVerticalStretch(100);
        sizePolicy6.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy6);
        widget->setMinimumSize(QSize(0, 0));
        verticalLayout_20 = new QVBoxLayout(widget);
        verticalLayout_20->setSpacing(6);
        verticalLayout_20->setObjectName(QStringLiteral("verticalLayout_20"));
        verticalLayout_20->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout_20->setContentsMargins(0, 0, 0, 10);
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        label_4 = new QLabel(widget);
        label_4->setObjectName(QStringLiteral("label_4"));
        QSizePolicy sizePolicy7(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(label_4->sizePolicy().hasHeightForWidth());
        label_4->setSizePolicy(sizePolicy7);
        label_4->setMinimumSize(QSize(0, 0));
        label_4->setMaximumSize(QSize(16777215, 1666666));

        horizontalLayout_8->addWidget(label_4);

        le_food_production_probability = new QLineEdit(widget);
        le_food_production_probability->setObjectName(QStringLiteral("le_food_production_probability"));

        horizontalLayout_8->addWidget(le_food_production_probability);


        verticalLayout_20->addLayout(horizontalLayout_8);

        horizontalLayout_24 = new QHBoxLayout();
        horizontalLayout_24->setObjectName(QStringLiteral("horizontalLayout_24"));
        label_15 = new QLabel(widget);
        label_15->setObjectName(QStringLiteral("label_15"));

        horizontalLayout_24->addWidget(label_15);

        le_produce_food_every_n_tick = new QLineEdit(widget);
        le_produce_food_every_n_tick->setObjectName(QStringLiteral("le_produce_food_every_n_tick"));

        horizontalLayout_24->addWidget(le_produce_food_every_n_tick);


        verticalLayout_20->addLayout(horizontalLayout_24);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        label_5 = new QLabel(widget);
        label_5->setObjectName(QStringLiteral("label_5"));

        horizontalLayout_9->addWidget(label_5);

        le_lifespan_multiplier = new QLineEdit(widget);
        le_lifespan_multiplier->setObjectName(QStringLiteral("le_lifespan_multiplier"));

        horizontalLayout_9->addWidget(le_lifespan_multiplier);


        verticalLayout_20->addLayout(horizontalLayout_9);

        verticalLayout_18 = new QVBoxLayout();
        verticalLayout_18->setObjectName(QStringLiteral("verticalLayout_18"));
        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QStringLiteral("horizontalLayout_13"));
        label_7 = new QLabel(widget);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout_13->addWidget(label_7);

        le_look_range = new QLineEdit(widget);
        le_look_range->setObjectName(QStringLiteral("le_look_range"));

        horizontalLayout_13->addWidget(le_look_range);


        verticalLayout_18->addLayout(horizontalLayout_13);


        verticalLayout_20->addLayout(verticalLayout_18);

        verticalLayout_14 = new QVBoxLayout();
        verticalLayout_14->setObjectName(QStringLiteral("verticalLayout_14"));
        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setObjectName(QStringLiteral("horizontalLayout_15"));
        label_9 = new QLabel(widget);
        label_9->setObjectName(QStringLiteral("label_9"));

        horizontalLayout_15->addWidget(label_9);

        le_auto_produce_n_food = new QLineEdit(widget);
        le_auto_produce_n_food->setObjectName(QStringLiteral("le_auto_produce_n_food"));

        horizontalLayout_15->addWidget(le_auto_produce_n_food);


        verticalLayout_14->addLayout(horizontalLayout_15);


        verticalLayout_20->addLayout(verticalLayout_14);

        horizontalLayout_42 = new QHBoxLayout();
        horizontalLayout_42->setObjectName(QStringLiteral("horizontalLayout_42"));
        label_44 = new QLabel(widget);
        label_44->setObjectName(QStringLiteral("label_44"));

        horizontalLayout_42->addWidget(label_44);

        le_auto_produce_food_every_n_tick = new QLineEdit(widget);
        le_auto_produce_food_every_n_tick->setObjectName(QStringLiteral("le_auto_produce_food_every_n_tick"));

        horizontalLayout_42->addWidget(le_auto_produce_food_every_n_tick);


        verticalLayout_20->addLayout(horizontalLayout_42);

        horizontalLayout_18 = new QHBoxLayout();
        horizontalLayout_18->setObjectName(QStringLiteral("horizontalLayout_18"));
        label_6 = new QLabel(widget);
        label_6->setObjectName(QStringLiteral("label_6"));

        horizontalLayout_18->addWidget(label_6);

        le_extra_reproduction_cost = new QLineEdit(widget);
        le_extra_reproduction_cost->setObjectName(QStringLiteral("le_extra_reproduction_cost"));

        horizontalLayout_18->addWidget(le_extra_reproduction_cost);


        verticalLayout_20->addLayout(horizontalLayout_18);

        cb_use_evolved_anatomy_mutation_rate = new QCheckBox(widget);
        cb_use_evolved_anatomy_mutation_rate->setObjectName(QStringLiteral("cb_use_evolved_anatomy_mutation_rate"));
        cb_use_evolved_anatomy_mutation_rate->setChecked(false);

        verticalLayout_20->addWidget(cb_use_evolved_anatomy_mutation_rate);

        mutation_rate_layout = new QVBoxLayout();
        mutation_rate_layout->setObjectName(QStringLiteral("mutation_rate_layout"));
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QStringLiteral("horizontalLayout_12"));
        lb_mutation_rate = new QLabel(widget);
        lb_mutation_rate->setObjectName(QStringLiteral("lb_mutation_rate"));
        lb_mutation_rate->setEnabled(true);

        horizontalLayout_12->addWidget(lb_mutation_rate);

        le_global_anatomy_mutation_rate = new QLineEdit(widget);
        le_global_anatomy_mutation_rate->setObjectName(QStringLiteral("le_global_anatomy_mutation_rate"));
        le_global_anatomy_mutation_rate->setEnabled(true);

        horizontalLayout_12->addWidget(le_global_anatomy_mutation_rate);


        mutation_rate_layout->addLayout(horizontalLayout_12);


        verticalLayout_20->addLayout(mutation_rate_layout);

        horizontalLayout_28 = new QHBoxLayout();
        horizontalLayout_28->setObjectName(QStringLiteral("horizontalLayout_28"));
        label_23 = new QLabel(widget);
        label_23->setObjectName(QStringLiteral("label_23"));

        horizontalLayout_28->addWidget(label_23);

        le_anatomy_mutation_rate_delimiter = new QLineEdit(widget);
        le_anatomy_mutation_rate_delimiter->setObjectName(QStringLiteral("le_anatomy_mutation_rate_delimiter"));

        horizontalLayout_28->addWidget(le_anatomy_mutation_rate_delimiter);


        verticalLayout_20->addLayout(horizontalLayout_28);

        cb_use_evolved_brain_mutation_rate = new QCheckBox(widget);
        cb_use_evolved_brain_mutation_rate->setObjectName(QStringLiteral("cb_use_evolved_brain_mutation_rate"));

        verticalLayout_20->addWidget(cb_use_evolved_brain_mutation_rate);

        horizontalLayout_27 = new QHBoxLayout();
        horizontalLayout_27->setObjectName(QStringLiteral("horizontalLayout_27"));
        label_20 = new QLabel(widget);
        label_20->setObjectName(QStringLiteral("label_20"));

        horizontalLayout_27->addWidget(label_20);

        le_global_brain_mutation_rate = new QLineEdit(widget);
        le_global_brain_mutation_rate->setObjectName(QStringLiteral("le_global_brain_mutation_rate"));

        horizontalLayout_27->addWidget(le_global_brain_mutation_rate);


        verticalLayout_20->addLayout(horizontalLayout_27);

        horizontalLayout_29 = new QHBoxLayout();
        horizontalLayout_29->setObjectName(QStringLiteral("horizontalLayout_29"));
        label_24 = new QLabel(widget);
        label_24->setObjectName(QStringLiteral("label_24"));

        horizontalLayout_29->addWidget(label_24);

        le_brain_mutation_rate_delimiter = new QLineEdit(widget);
        le_brain_mutation_rate_delimiter->setObjectName(QStringLiteral("le_brain_mutation_rate_delimiter"));

        horizontalLayout_29->addWidget(le_brain_mutation_rate_delimiter);


        verticalLayout_20->addLayout(horizontalLayout_29);

        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setObjectName(QStringLiteral("horizontalLayout_19"));
        label_21 = new QLabel(widget);
        label_21->setObjectName(QStringLiteral("label_21"));

        horizontalLayout_19->addWidget(label_21);

        le_killer_damage_amount = new QLineEdit(widget);
        le_killer_damage_amount->setObjectName(QStringLiteral("le_killer_damage_amount"));

        horizontalLayout_19->addWidget(le_killer_damage_amount);


        verticalLayout_20->addLayout(horizontalLayout_19);

        verticalLayout_17 = new QVBoxLayout();
        verticalLayout_17->setObjectName(QStringLiteral("verticalLayout_17"));
        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QStringLiteral("horizontalLayout_16"));
        label_10 = new QLabel(widget);
        label_10->setObjectName(QStringLiteral("label_10"));

        horizontalLayout_16->addWidget(label_10);

        le_add = new QLineEdit(widget);
        le_add->setObjectName(QStringLiteral("le_add"));

        horizontalLayout_16->addWidget(le_add);

        label_11 = new QLabel(widget);
        label_11->setObjectName(QStringLiteral("label_11"));

        horizontalLayout_16->addWidget(label_11);

        le_change = new QLineEdit(widget);
        le_change->setObjectName(QStringLiteral("le_change"));

        horizontalLayout_16->addWidget(le_change);

        label_12 = new QLabel(widget);
        label_12->setObjectName(QStringLiteral("label_12"));

        horizontalLayout_16->addWidget(label_12);

        le_remove = new QLineEdit(widget);
        le_remove->setObjectName(QStringLiteral("le_remove"));

        horizontalLayout_16->addWidget(le_remove);


        verticalLayout_17->addLayout(horizontalLayout_16);


        verticalLayout_20->addLayout(verticalLayout_17);

        cb_runtime_rotation_enabled = new QCheckBox(widget);
        cb_runtime_rotation_enabled->setObjectName(QStringLiteral("cb_runtime_rotation_enabled"));

        verticalLayout_20->addWidget(cb_runtime_rotation_enabled);

        cb_reproducing_rotation_enabled = new QCheckBox(widget);
        cb_reproducing_rotation_enabled->setObjectName(QStringLiteral("cb_reproducing_rotation_enabled"));
        cb_reproducing_rotation_enabled->setChecked(true);

        verticalLayout_20->addWidget(cb_reproducing_rotation_enabled);

        cb_on_touch_kill = new QCheckBox(widget);
        cb_on_touch_kill->setObjectName(QStringLiteral("cb_on_touch_kill"));
        cb_on_touch_kill->setChecked(false);

        verticalLayout_20->addWidget(cb_on_touch_kill);

        cb_movers_can_produce_food = new QCheckBox(widget);
        cb_movers_can_produce_food->setObjectName(QStringLiteral("cb_movers_can_produce_food"));

        verticalLayout_20->addWidget(cb_movers_can_produce_food);

        cb_food_blocks_reproduction = new QCheckBox(widget);
        cb_food_blocks_reproduction->setObjectName(QStringLiteral("cb_food_blocks_reproduction"));
        cb_food_blocks_reproduction->setEnabled(true);
        cb_food_blocks_reproduction->setChecked(true);

        verticalLayout_20->addWidget(cb_food_blocks_reproduction);

        cb_food_blocks_movement = new QCheckBox(widget);
        cb_food_blocks_movement->setObjectName(QStringLiteral("cb_food_blocks_movement"));
        cb_food_blocks_movement->setChecked(true);

        verticalLayout_20->addWidget(cb_food_blocks_movement);

        verticalLayout_26 = new QVBoxLayout();
        verticalLayout_26->setObjectName(QStringLiteral("verticalLayout_26"));
        horizontalLayout_25 = new QHBoxLayout();
        horizontalLayout_25->setObjectName(QStringLiteral("horizontalLayout_25"));
        label_17 = new QLabel(widget);
        label_17->setObjectName(QStringLiteral("label_17"));

        horizontalLayout_25->addWidget(label_17);

        le_min_reproduction_distance = new QLineEdit(widget);
        le_min_reproduction_distance->setObjectName(QStringLiteral("le_min_reproduction_distance"));

        horizontalLayout_25->addWidget(le_min_reproduction_distance);


        verticalLayout_26->addLayout(horizontalLayout_25);

        horizontalLayout_26 = new QHBoxLayout();
        horizontalLayout_26->setObjectName(QStringLiteral("horizontalLayout_26"));
        label_16 = new QLabel(widget);
        label_16->setObjectName(QStringLiteral("label_16"));

        horizontalLayout_26->addWidget(label_16);

        le_max_reproduction_distance = new QLineEdit(widget);
        le_max_reproduction_distance->setObjectName(QStringLiteral("le_max_reproduction_distance"));

        horizontalLayout_26->addWidget(le_max_reproduction_distance);


        verticalLayout_26->addLayout(horizontalLayout_26);

        cb_fix_reproduction_distance = new QCheckBox(widget);
        cb_fix_reproduction_distance->setObjectName(QStringLiteral("cb_fix_reproduction_distance"));

        verticalLayout_26->addWidget(cb_fix_reproduction_distance);

        cb_self_organism_blocks_block_sight = new QCheckBox(widget);
        cb_self_organism_blocks_block_sight->setObjectName(QStringLiteral("cb_self_organism_blocks_block_sight"));

        verticalLayout_26->addWidget(cb_self_organism_blocks_block_sight);


        verticalLayout_20->addLayout(verticalLayout_26);

        verticalLayout_34 = new QVBoxLayout();
        verticalLayout_34->setObjectName(QStringLiteral("verticalLayout_34"));
        cb_set_fixed_move_range = new QCheckBox(widget);
        cb_set_fixed_move_range->setObjectName(QStringLiteral("cb_set_fixed_move_range"));

        verticalLayout_34->addWidget(cb_set_fixed_move_range);

        horizontalLayout_39 = new QHBoxLayout();
        horizontalLayout_39->setObjectName(QStringLiteral("horizontalLayout_39"));
        label_41 = new QLabel(widget);
        label_41->setObjectName(QStringLiteral("label_41"));

        horizontalLayout_39->addWidget(label_41);

        le_min_move_range = new QLineEdit(widget);
        le_min_move_range->setObjectName(QStringLiteral("le_min_move_range"));

        horizontalLayout_39->addWidget(le_min_move_range);


        verticalLayout_34->addLayout(horizontalLayout_39);

        horizontalLayout_40 = new QHBoxLayout();
        horizontalLayout_40->setObjectName(QStringLiteral("horizontalLayout_40"));
        label_40 = new QLabel(widget);
        label_40->setObjectName(QStringLiteral("label_40"));

        horizontalLayout_40->addWidget(label_40);

        le_max_move_range = new QLineEdit(widget);
        le_max_move_range->setObjectName(QStringLiteral("le_max_move_range"));

        horizontalLayout_40->addWidget(le_max_move_range);


        verticalLayout_34->addLayout(horizontalLayout_40);

        horizontalLayout_38 = new QHBoxLayout();
        horizontalLayout_38->setObjectName(QStringLiteral("horizontalLayout_38"));
        label_42 = new QLabel(widget);
        label_42->setObjectName(QStringLiteral("label_42"));

        horizontalLayout_38->addWidget(label_42);

        le_move_range_delimiter = new QLineEdit(widget);
        le_move_range_delimiter->setObjectName(QStringLiteral("le_move_range_delimiter"));

        horizontalLayout_38->addWidget(le_move_range_delimiter);


        verticalLayout_34->addLayout(horizontalLayout_38);


        verticalLayout_20->addLayout(verticalLayout_34);

        cb_failed_reproduction_eats_food = new QCheckBox(widget);
        cb_failed_reproduction_eats_food->setObjectName(QStringLiteral("cb_failed_reproduction_eats_food"));

        verticalLayout_20->addWidget(cb_failed_reproduction_eats_food);

        cb_rotate_every_move_tick = new QCheckBox(widget);
        cb_rotate_every_move_tick->setObjectName(QStringLiteral("cb_rotate_every_move_tick"));

        verticalLayout_20->addWidget(cb_rotate_every_move_tick);

        cb_multiply_food_production_prob = new QCheckBox(widget);
        cb_multiply_food_production_prob->setObjectName(QStringLiteral("cb_multiply_food_production_prob"));

        verticalLayout_20->addWidget(cb_multiply_food_production_prob);

        cb_simplified_food_production = new QCheckBox(widget);
        cb_simplified_food_production->setObjectName(QStringLiteral("cb_simplified_food_production"));

        verticalLayout_20->addWidget(cb_simplified_food_production);

        cb_stop_when_one_food_generated = new QCheckBox(widget);
        cb_stop_when_one_food_generated->setObjectName(QStringLiteral("cb_stop_when_one_food_generated"));

        verticalLayout_20->addWidget(cb_stop_when_one_food_generated);

        cb_eat_then_produce = new QCheckBox(widget);
        cb_eat_then_produce->setObjectName(QStringLiteral("cb_eat_then_produce"));

        verticalLayout_20->addWidget(cb_eat_then_produce);


        verticalLayout_9->addWidget(widget);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        horizontalLayout_7->addWidget(scrollArea_2);

        Tabs->addTab(evolution_controls_tab, QString());
        statistics_tab = new QWidget();
        statistics_tab->setObjectName(QStringLiteral("statistics_tab"));
        verticalLayout_3 = new QVBoxLayout(statistics_tab);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        scrollArea_4 = new QScrollArea(statistics_tab);
        scrollArea_4->setObjectName(QStringLiteral("scrollArea_4"));
        scrollArea_4->setMinimumSize(QSize(0, 0));
        scrollArea_4->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QStringLiteral("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 518, 217));
        QSizePolicy sizePolicy8(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(scrollAreaWidgetContents_4->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_4->setSizePolicy(sizePolicy8);
        verticalLayout_24 = new QVBoxLayout(scrollAreaWidgetContents_4);
        verticalLayout_24->setObjectName(QStringLiteral("verticalLayout_24"));
        widget_5 = new QWidget(scrollAreaWidgetContents_4);
        widget_5->setObjectName(QStringLiteral("widget_5"));
        sizePolicy7.setHeightForWidth(widget_5->sizePolicy().hasHeightForWidth());
        widget_5->setSizePolicy(sizePolicy7);
        widget_5->setMinimumSize(QSize(0, 0));
        verticalLayout_37 = new QVBoxLayout(widget_5);
        verticalLayout_37->setObjectName(QStringLiteral("verticalLayout_37"));
        verticalLayout_37->setContentsMargins(0, 0, 0, 0);
        tb_open_statisctics = new QPushButton(widget_5);
        tb_open_statisctics->setObjectName(QStringLiteral("tb_open_statisctics"));
        tb_open_statisctics->setCheckable(true);

        verticalLayout_37->addWidget(tb_open_statisctics);

        cb_statistics_always_on_top = new QCheckBox(widget_5);
        cb_statistics_always_on_top->setObjectName(QStringLiteral("cb_statistics_always_on_top"));

        verticalLayout_37->addWidget(cb_statistics_always_on_top);


        verticalLayout_24->addWidget(widget_5);

        scrollArea_4->setWidget(scrollAreaWidgetContents_4);

        verticalLayout_3->addWidget(scrollArea_4);

        Tabs->addTab(statistics_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName(QStringLiteral("settings_tab"));
        verticalLayout_7 = new QVBoxLayout(settings_tab);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(settings_tab);
        scrollArea->setObjectName(QStringLiteral("scrollArea"));
        scrollArea->setLineWidth(0);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 504, 610));
        QSizePolicy sizePolicy9(QSizePolicy::Ignored, QSizePolicy::Preferred);
        sizePolicy9.setHorizontalStretch(0);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(scrollAreaWidgetContents->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents->setSizePolicy(sizePolicy9);
        verticalLayout_5 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        widget_2 = new QWidget(scrollAreaWidgetContents);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        widget_2->setMinimumSize(QSize(0, 0));
        widget_2->setBaseSize(QSize(0, 0));
        verticalLayout_19 = new QVBoxLayout(widget_2);
        verticalLayout_19->setSpacing(2);
        verticalLayout_19->setObjectName(QStringLiteral("verticalLayout_19"));
        verticalLayout_19->setContentsMargins(9, 9, 9, 9);
        verticalLayout_23 = new QVBoxLayout();
        verticalLayout_23->setObjectName(QStringLiteral("verticalLayout_23"));
        rb_single_thread_mode = new QRadioButton(widget_2);
        simulation_modes = new QButtonGroup(MainWindow);
        simulation_modes->setObjectName(QStringLiteral("simulation_modes"));
        simulation_modes->addButton(rb_single_thread_mode);
        rb_single_thread_mode->setObjectName(QStringLiteral("rb_single_thread_mode"));
        rb_single_thread_mode->setEnabled(false);
        rb_single_thread_mode->setChecked(true);

        verticalLayout_23->addWidget(rb_single_thread_mode);

        rb_partial_multi_thread_mode = new QRadioButton(widget_2);
        simulation_modes->addButton(rb_partial_multi_thread_mode);
        rb_partial_multi_thread_mode->setObjectName(QStringLiteral("rb_partial_multi_thread_mode"));
        rb_partial_multi_thread_mode->setEnabled(false);

        verticalLayout_23->addWidget(rb_partial_multi_thread_mode);

        rb_multi_thread_mode = new QRadioButton(widget_2);
        simulation_modes->addButton(rb_multi_thread_mode);
        rb_multi_thread_mode->setObjectName(QStringLiteral("rb_multi_thread_mode"));
        rb_multi_thread_mode->setEnabled(false);
        rb_multi_thread_mode->setChecked(false);

        verticalLayout_23->addWidget(rb_multi_thread_mode);

        rb_cuda_mode = new QRadioButton(widget_2);
        simulation_modes->addButton(rb_cuda_mode);
        rb_cuda_mode->setObjectName(QStringLiteral("rb_cuda_mode"));
        rb_cuda_mode->setEnabled(false);

        verticalLayout_23->addWidget(rb_cuda_mode);


        verticalLayout_19->addLayout(verticalLayout_23);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        label = new QLabel(widget_2);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout_4->addWidget(label);

        le_num_threads = new QLineEdit(widget_2);
        le_num_threads->setObjectName(QStringLiteral("le_num_threads"));
        le_num_threads->setEnabled(true);

        horizontalLayout_4->addWidget(le_num_threads);


        verticalLayout_19->addLayout(horizontalLayout_4);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        label_19 = new QLabel(widget_2);
        label_19->setObjectName(QStringLiteral("label_19"));

        horizontalLayout_2->addWidget(label_19);

        le_float_number_precision = new QLineEdit(widget_2);
        le_float_number_precision->setObjectName(QStringLiteral("le_float_number_precision"));

        horizontalLayout_2->addWidget(le_float_number_precision);


        verticalLayout_19->addLayout(horizontalLayout_2);

        cb_disable_warnings = new QCheckBox(widget_2);
        cb_disable_warnings->setObjectName(QStringLiteral("cb_disable_warnings"));

        verticalLayout_19->addWidget(cb_disable_warnings);

        cb_wait_for_engine_to_stop = new QCheckBox(widget_2);
        cb_wait_for_engine_to_stop->setObjectName(QStringLiteral("cb_wait_for_engine_to_stop"));

        verticalLayout_19->addWidget(cb_wait_for_engine_to_stop);

        horizontalLayout_30 = new QHBoxLayout();
        horizontalLayout_30->setObjectName(QStringLiteral("horizontalLayout_30"));
        label_25 = new QLabel(widget_2);
        label_25->setObjectName(QStringLiteral("label_25"));

        horizontalLayout_30->addWidget(label_25);

        le_font_size = new QLineEdit(widget_2);
        le_font_size->setObjectName(QStringLiteral("le_font_size"));

        horizontalLayout_30->addWidget(le_font_size);


        verticalLayout_19->addLayout(horizontalLayout_30);

        cb_simplified_rendering = new QCheckBox(widget_2);
        cb_simplified_rendering->setObjectName(QStringLiteral("cb_simplified_rendering"));

        verticalLayout_19->addWidget(cb_simplified_rendering);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        label_46 = new QLabel(widget_2);
        label_46->setObjectName(QStringLiteral("label_46"));

        horizontalLayout_10->addWidget(label_46);

        le_update_info_every_n_milliseconds = new QLineEdit(widget_2);
        le_update_info_every_n_milliseconds->setObjectName(QStringLiteral("le_update_info_every_n_milliseconds"));

        horizontalLayout_10->addWidget(le_update_info_every_n_milliseconds);


        verticalLayout_19->addLayout(horizontalLayout_10);

        cb_synchronise_info_with_window = new QCheckBox(widget_2);
        cb_synchronise_info_with_window->setObjectName(QStringLiteral("cb_synchronise_info_with_window"));

        verticalLayout_19->addWidget(cb_synchronise_info_with_window);

        cb_use_nvidia_for_image_generation = new QCheckBox(widget_2);
        cb_use_nvidia_for_image_generation->setObjectName(QStringLiteral("cb_use_nvidia_for_image_generation"));

        verticalLayout_19->addWidget(cb_use_nvidia_for_image_generation);

        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setObjectName(QStringLiteral("horizontalLayout_14"));
        label_47 = new QLabel(widget_2);
        label_47->setObjectName(QStringLiteral("label_47"));

        horizontalLayout_14->addWidget(label_47);

        le_menu_height = new QLineEdit(widget_2);
        le_menu_height->setObjectName(QStringLiteral("le_menu_height"));

        horizontalLayout_14->addWidget(le_menu_height);


        verticalLayout_19->addLayout(horizontalLayout_14);

        verticalSpacer_2 = new QSpacerItem(20, 15, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_19->addItem(verticalSpacer_2);

        label_22 = new QLabel(widget_2);
        label_22->setObjectName(QStringLiteral("label_22"));

        verticalLayout_19->addWidget(label_22);

        horizontalLayout_20 = new QHBoxLayout();
        horizontalLayout_20->setObjectName(QStringLiteral("horizontalLayout_20"));
        label_26 = new QLabel(widget_2);
        label_26->setObjectName(QStringLiteral("label_26"));

        horizontalLayout_20->addWidget(label_26);

        le_perlin_octaves = new QLineEdit(widget_2);
        le_perlin_octaves->setObjectName(QStringLiteral("le_perlin_octaves"));

        horizontalLayout_20->addWidget(le_perlin_octaves);


        verticalLayout_19->addLayout(horizontalLayout_20);

        horizontalLayout_31 = new QHBoxLayout();
        horizontalLayout_31->setObjectName(QStringLiteral("horizontalLayout_31"));
        label_27 = new QLabel(widget_2);
        label_27->setObjectName(QStringLiteral("label_27"));

        horizontalLayout_31->addWidget(label_27);

        le_perlin_persistence = new QLineEdit(widget_2);
        le_perlin_persistence->setObjectName(QStringLiteral("le_perlin_persistence"));

        horizontalLayout_31->addWidget(le_perlin_persistence);


        verticalLayout_19->addLayout(horizontalLayout_31);

        horizontalLayout_32 = new QHBoxLayout();
        horizontalLayout_32->setObjectName(QStringLiteral("horizontalLayout_32"));
        label_28 = new QLabel(widget_2);
        label_28->setObjectName(QStringLiteral("label_28"));

        horizontalLayout_32->addWidget(label_28);

        le_perlin_upper_bound = new QLineEdit(widget_2);
        le_perlin_upper_bound->setObjectName(QStringLiteral("le_perlin_upper_bound"));

        horizontalLayout_32->addWidget(le_perlin_upper_bound);


        verticalLayout_19->addLayout(horizontalLayout_32);

        horizontalLayout_33 = new QHBoxLayout();
        horizontalLayout_33->setObjectName(QStringLiteral("horizontalLayout_33"));
        label_29 = new QLabel(widget_2);
        label_29->setObjectName(QStringLiteral("label_29"));

        horizontalLayout_33->addWidget(label_29);

        le_perlin_lower_bound = new QLineEdit(widget_2);
        le_perlin_lower_bound->setObjectName(QStringLiteral("le_perlin_lower_bound"));

        horizontalLayout_33->addWidget(le_perlin_lower_bound);


        verticalLayout_19->addLayout(horizontalLayout_33);

        horizontalLayout_34 = new QHBoxLayout();
        horizontalLayout_34->setObjectName(QStringLiteral("horizontalLayout_34"));
        label_30 = new QLabel(widget_2);
        label_30->setObjectName(QStringLiteral("label_30"));

        horizontalLayout_34->addWidget(label_30);

        le_perlin_x_modifier = new QLineEdit(widget_2);
        le_perlin_x_modifier->setObjectName(QStringLiteral("le_perlin_x_modifier"));

        horizontalLayout_34->addWidget(le_perlin_x_modifier);


        verticalLayout_19->addLayout(horizontalLayout_34);

        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setObjectName(QStringLiteral("horizontalLayout_17"));
        label_31 = new QLabel(widget_2);
        label_31->setObjectName(QStringLiteral("label_31"));

        horizontalLayout_17->addWidget(label_31);

        le_perlin_y_modifier = new QLineEdit(widget_2);
        le_perlin_y_modifier->setObjectName(QStringLiteral("le_perlin_y_modifier"));

        horizontalLayout_17->addWidget(le_perlin_y_modifier);


        verticalLayout_19->addLayout(horizontalLayout_17);


        verticalLayout_5->addWidget(widget_2);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_7->addWidget(scrollArea);

        Tabs->addTab(settings_tab, QString());

        verticalLayout_4->addWidget(Tabs);


        horizontalLayout->addWidget(frame_3);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 4);
        frame_3->raise();
        frame_2->raise();

        verticalLayout_2->addWidget(menu_frame);

        verticalLayout_2->setStretch(0, 3);
        verticalLayout_2->setStretch(1, 1);
#ifndef QT_NO_SHORTCUT
        lb_fps->setBuddy(le_fps);
#endif // QT_NO_SHORTCUT

        retranslateUi(MainWindow);
        QObject::connect(b_reset, SIGNAL(clicked()), MainWindow, SLOT(b_reset_slot()));
        QObject::connect(cb_synchronise_sim_and_win, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_synchronise_simulation_and_window_slot(bool)));
        QObject::connect(tb_stoprender, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_stoprender_slot(bool)));
        QObject::connect(rb_wall, SIGNAL(clicked()), MainWindow, SLOT(rb_wall_slot()));
        QObject::connect(b_reset_view, SIGNAL(clicked()), MainWindow, SLOT(b_reset_view_slot()));
        QObject::connect(rb_kill, SIGNAL(clicked()), MainWindow, SLOT(rb_kill_slot()));
        QObject::connect(le_fps, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_fps_slot()));
        QObject::connect(b_clear, SIGNAL(clicked()), MainWindow, SLOT(b_clear_slot()));
        QObject::connect(tb_pause, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_pause_slot(bool)));
        QObject::connect(le_sps, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_sps_slot()));
        QObject::connect(b_pass_one_tick, SIGNAL(clicked()), MainWindow, SLOT(b_pass_one_tick_slot()));
        QObject::connect(b_kill_all_organisms_slot, SIGNAL(clicked()), MainWindow, SLOT(b_kill_all_organisms_slot()));
        QObject::connect(rb_food, SIGNAL(clicked()), MainWindow, SLOT(rb_food_slot()));
        QObject::connect(cb_movers_can_produce_food, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_movers_can_produce_food_slot(bool)));
        QObject::connect(le_simulation_width, SIGNAL(returnPressed()), MainWindow, SLOT(le_simulation_width_slot()));
        QObject::connect(b_load_world, SIGNAL(clicked()), MainWindow, SLOT(b_load_world_slot()));
        QObject::connect(cb_reset_on_total_extinction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_reset_on_total_extinction_slot(bool)));
        QObject::connect(cb_fill_window, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_fill_window_slot(bool)));
        QObject::connect(rb_partial_multi_thread_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_partial_multi_thread_slot()));
        QObject::connect(b_save_world, SIGNAL(clicked()), MainWindow, SLOT(b_save_world_slot()));
        QObject::connect(rb_single_thread_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_single_thread_slot()));
        QObject::connect(cb_pause_on_total_extinction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_pause_on_total_extinction_slot(bool)));
        QObject::connect(b_generate_random_walls, SIGNAL(clicked()), MainWindow, SLOT(b_generate_random_walls_slot()));
        QObject::connect(cb_disable_warnings, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_disable_warnings_slot(bool)));
        QObject::connect(le_auto_produce_n_food, SIGNAL(returnPressed()), MainWindow, SLOT(le_auto_food_drop_rate_slot()));
        QObject::connect(le_num_threads, SIGNAL(returnPressed()), MainWindow, SLOT(le_num_threads_slot()));
        QObject::connect(le_remove, SIGNAL(returnPressed()), MainWindow, SLOT(le_remove_cell_slot()));
        QObject::connect(le_min_reproduction_distance, SIGNAL(returnPressed()), MainWindow, SLOT(le_min_reproducing_distance_slot()));
        QObject::connect(le_change, SIGNAL(returnPressed()), MainWindow, SLOT(le_change_cell_slot()));
        QObject::connect(le_produce_food_every_n_tick, SIGNAL(returnPressed()), MainWindow, SLOT(le_produce_food_every_n_slot()));
        QObject::connect(le_look_range, SIGNAL(returnPressed()), MainWindow, SLOT(le_look_range_slot()));
        QObject::connect(cb_runtime_rotation_enabled, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_runtime_rotation_enabled_slot(bool)));
        QObject::connect(cb_generate_random_walls_on_reset, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_generate_random_walls_on_reset_slot(bool)));
        QObject::connect(le_float_number_precision, SIGNAL(returnPressed()), MainWindow, SLOT(le_float_number_precision_slot()));
        QObject::connect(le_anatomy_mutation_rate_delimiter, SIGNAL(returnPressed()), MainWindow, SLOT(le_anatomy_mutation_rate_delimiter_slot()));
        QObject::connect(rb_cuda_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_cuda_slot()));
        QObject::connect(le_global_anatomy_mutation_rate, SIGNAL(returnPressed()), MainWindow, SLOT(le_global_anatomy_mutation_rate_slot()));
        QObject::connect(le_max_reproduction_distance, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_reproducing_distance_slot()));
        QObject::connect(cb_fix_reproduction_distance, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_fix_reproduction_distance_slot(bool)));
        QObject::connect(cb_use_evolved_anatomy_mutation_rate, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_use_evolved_anatomy_mutation_rate_slot(bool)));
        QObject::connect(le_killer_damage_amount, SIGNAL(returnPressed()), MainWindow, SLOT(le_killer_damage_amount_slot()));
        QObject::connect(cb_on_touch_kill, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_on_touch_kill_slot(bool)));
        QObject::connect(b_resize_and_reset, SIGNAL(clicked()), MainWindow, SLOT(b_resize_and_reset_slot()));
        QObject::connect(le_food_production_probability, SIGNAL(returnPressed()), MainWindow, SLOT(le_food_production_probability_slot()));
        QObject::connect(le_extra_reproduction_cost, SIGNAL(returnPressed()), MainWindow, SLOT(le_extra_reproduction_cost_slot()));
        QObject::connect(le_lifespan_multiplier, SIGNAL(returnPressed()), MainWindow, SLOT(le_lifespan_multiplier_slot()));
        QObject::connect(le_brain_mutation_rate_delimiter, SIGNAL(returnPressed()), MainWindow, SLOT(le_brain_mutation_rate_delimiter_slot()));
        QObject::connect(le_global_brain_mutation_rate, SIGNAL(returnPressed()), MainWindow, SLOT(le_global_brain_mutation_rate_slot()));
        QObject::connect(le_cell_size, SIGNAL(returnPressed()), MainWindow, SLOT(le_cell_size_slot()));
        QObject::connect(cb_use_evolved_brain_mutation_rate, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_use_evolved_brain_mutation_rate_slot(bool)));
        QObject::connect(cb_self_organism_blocks_block_sight, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_self_organism_blocks_block_sight_slot(bool)));
        QObject::connect(le_add, SIGNAL(returnPressed()), MainWindow, SLOT(le_add_cell_slot()));
        QObject::connect(le_simulation_height, SIGNAL(returnPressed()), MainWindow, SLOT(le_simulation_height_slot()));
        QObject::connect(le_max_organisms, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_organisms_slot()));
        QObject::connect(rb_multi_thread_mode, SIGNAL(clicked()), MainWindow, SLOT(rb_multi_thread_slot()));
        QObject::connect(cb_clear_walls_on_reset, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_clear_walls_on_reset_slot(bool)));
        QObject::connect(cb_food_blocks_reproduction, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_food_blocks_reproduction_slot(bool)));
        QObject::connect(cb_reproducing_rotation_enabled, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_reproduction_rotation_enabled_slot(bool)));
        QObject::connect(b_clear_all_walls, SIGNAL(clicked()), MainWindow, SLOT(b_clear_all_walls_slot()));
        QObject::connect(le_font_size, SIGNAL(returnPressed()), MainWindow, SLOT(le_font_size_slot()));
        QObject::connect(cb_set_fixed_move_range, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_set_fixed_move_range_slot(bool)));
        QObject::connect(le_max_move_range, SIGNAL(returnPressed()), MainWindow, SLOT(le_max_move_range_slot()));
        QObject::connect(le_min_move_range, SIGNAL(returnPressed()), MainWindow, SLOT(le_min_move_range_slot()));
        QObject::connect(le_move_range_delimiter, SIGNAL(returnPressed()), MainWindow, SLOT(le_move_range_delimiter_slot()));
        QObject::connect(le_brush_size, SIGNAL(returnPressed()), MainWindow, SLOT(le_brush_size_slot()));
        QObject::connect(cb_failed_reproduction_eats_food, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_failed_reproduction_eats_food_slot(bool)));
        QObject::connect(le_auto_produce_food_every_n_tick, SIGNAL(returnPressed()), MainWindow, SLOT(le_auto_produce_food_every_n_tick_slot()));
        QObject::connect(table_organism_block_parameters, SIGNAL(cellChanged(int,int)), MainWindow, SLOT(table_cell_changed_slot(int,int)));
        QObject::connect(cb_wait_for_engine_to_stop, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_wait_for_engine_to_stop_slot(bool)));
        QObject::connect(cb_rotate_every_move_tick, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_rotate_every_move_tick_slot(bool)));
        QObject::connect(cb_simplified_rendering, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_simplified_rendering_slot(bool)));
        QObject::connect(cb_multiply_food_production_prob, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_multiply_food_production_prob_slot(bool)));
        QObject::connect(cb_simplified_food_production, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_simplified_food_production_slot(bool)));
        QObject::connect(cb_stop_when_one_food_generated, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_stop_when_one_food_generated(bool)));
        QObject::connect(cb_synchronise_info_with_window, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_synchronise_info_with_window_slot(bool)));
        QObject::connect(le_update_info_every_n_milliseconds, SIGNAL(returnPressed()), MainWindow, SLOT(le_update_info_every_n_milliseconds_slot()));
        QObject::connect(cb_use_nvidia_for_image_generation, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_use_nvidia_for_image_generation_slot(bool)));
        QObject::connect(cb_eat_then_produce, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_eat_then_produce_slot(bool)));
        QObject::connect(le_menu_height, SIGNAL(returnPressed()), MainWindow, SLOT(le_menu_height_slot()));
        QObject::connect(tb_open_statisctics, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_open_statistics_slot(bool)));
        QObject::connect(cb_statistics_always_on_top, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_statistics_always_on_top_slot(bool)));
        QObject::connect(tb_open_organism_editor, SIGNAL(toggled(bool)), MainWindow, SLOT(tb_open_organism_editor_slot(bool)));
        QObject::connect(cb_editor_always_on_top, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_editor_always_on_top_slot(bool)));
        QObject::connect(le_perlin_lower_bound, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_lower_bound_slot()));
        QObject::connect(le_perlin_octaves, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_octaves_slot()));
        QObject::connect(le_perlin_persistence, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_persistence_slot()));
        QObject::connect(le_perlin_upper_bound, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_upper_bound_slot()));
        QObject::connect(le_perlin_x_modifier, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_x_modifier_slot()));
        QObject::connect(le_perlin_y_modifier, SIGNAL(returnPressed()), MainWindow, SLOT(le_perlin_y_modifier_slot()));
        QObject::connect(cb_food_blocks_movement, SIGNAL(toggled(bool)), MainWindow, SLOT(cb_food_blocks_movement_slot(bool)));

        Tabs->setCurrentIndex(3);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        lb_fps->setText(QApplication::translate("MainWindow", "fps:", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "Set max fps:", Q_NULLPTR));
        le_fps->setText(QApplication::translate("MainWindow", "60", Q_NULLPTR));
        lb_sps->setText(QApplication::translate("MainWindow", "sps:", Q_NULLPTR));
        label_3->setText(QApplication::translate("MainWindow", "Set max sps:", Q_NULLPTR));
        le_sps->setText(QApplication::translate("MainWindow", "-1", Q_NULLPTR));
        b_reset->setText(QApplication::translate("MainWindow", "Reset", Q_NULLPTR));
        b_reset_view->setText(QApplication::translate("MainWindow", "Reset view", Q_NULLPTR));
        b_pass_one_tick->setText(QApplication::translate("MainWindow", "Pass one tick", Q_NULLPTR));
        tb_stoprender->setText(QApplication::translate("MainWindow", "Stop render", Q_NULLPTR));
        b_kill_all_organisms_slot->setText(QApplication::translate("MainWindow", "Kill all", Q_NULLPTR));
        tb_pause->setText(QApplication::translate("MainWindow", "Pause", Q_NULLPTR));
        b_clear->setText(QApplication::translate("MainWindow", "Clear", Q_NULLPTR));
        rb_food->setText(QApplication::translate("MainWindow", "Food mode", Q_NULLPTR));
        rb_kill->setText(QApplication::translate("MainWindow", "Kill mode", Q_NULLPTR));
        rb_wall->setText(QApplication::translate("MainWindow", "Wall mode", Q_NULLPTR));
        rb_null_button->setText(QString());
        label_43->setText(QApplication::translate("MainWindow", "Brush size:", Q_NULLPTR));
        cb_synchronise_sim_and_win->setText(QApplication::translate("MainWindow", "Synchronise simulation and window", Q_NULLPTR));
        textEdit->setHtml(QApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This is c++ implementation of an original The Life Engine (https://thelifeengine.net/) made in javascript (https://github.com/MaxRobinsonTheGreat/LifeEngine).</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This version will (in the future) feature everything that the original version has, but I (maybe) will also add some new features.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This version is in alp"
                        "ha stage, so many things are not implemeted and will probably change.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Important Information:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">*sps is the number of simulation ticks per second.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">*setting max fps/sps/num organisms to &lt;0 will enable unlimited mode.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">*checking &quot;Fix reproducing distance&quot; will set reproducing distance to min value.</p>\n"
"<p style=\" margin-top:0px; margi"
                        "n-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">*to hide menu, make sure that you don't enter text, put cursor on simulation grid, and press the keyboard button &quot;M&quot;</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This project's Github: https://github.com/SuperSpaceEye/TheLifeEngineCpp</p></body></html>", Q_NULLPTR));
        Tabs->setTabText(Tabs->indexOf(about_tab), QApplication::translate("MainWindow", "About", Q_NULLPTR));
        tb_open_organism_editor->setText(QApplication::translate("MainWindow", "Open Organism Editor Window", Q_NULLPTR));
        cb_editor_always_on_top->setText(QApplication::translate("MainWindow", "Always on top", Q_NULLPTR));
        Tabs->setTabText(Tabs->indexOf(editor_tab), QApplication::translate("MainWindow", "Editor", Q_NULLPTR));
        label_8->setText(QApplication::translate("MainWindow", "Start cell size:", Q_NULLPTR));
        le_cell_size->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        cb_fill_window->setText(QApplication::translate("MainWindow", "Fill window", Q_NULLPTR));
        label_13->setText(QApplication::translate("MainWindow", "Width:", Q_NULLPTR));
        le_simulation_width->setText(QApplication::translate("MainWindow", "600", Q_NULLPTR));
        label_14->setText(QApplication::translate("MainWindow", "Height:", Q_NULLPTR));
        le_simulation_height->setText(QApplication::translate("MainWindow", "600", Q_NULLPTR));
        b_resize_and_reset->setText(QApplication::translate("MainWindow", "Resize and Reset", Q_NULLPTR));
        cb_reset_on_total_extinction->setText(QApplication::translate("MainWindow", "Reset on total extinction", Q_NULLPTR));
        lb_auto_reset_count->setText(QApplication::translate("MainWindow", "Auto reset count: 0", Q_NULLPTR));
        cb_pause_on_total_extinction->setText(QApplication::translate("MainWindow", "Pause on total extinction", Q_NULLPTR));
        label_18->setText(QApplication::translate("MainWindow", "Max organisms:", Q_NULLPTR));
        le_max_organisms->setText(QApplication::translate("MainWindow", "-1", Q_NULLPTR));
        b_generate_random_walls->setText(QApplication::translate("MainWindow", "Generate random walls", Q_NULLPTR));
        cb_generate_random_walls_on_reset->setText(QApplication::translate("MainWindow", "Generate random walls on reset", Q_NULLPTR));
        b_clear_all_walls->setText(QApplication::translate("MainWindow", "Clear all walls", Q_NULLPTR));
        cb_clear_walls_on_reset->setText(QApplication::translate("MainWindow", "Clear walls on reset", Q_NULLPTR));
        b_save_world->setText(QApplication::translate("MainWindow", "Save world", Q_NULLPTR));
        b_load_world->setText(QApplication::translate("MainWindow", "Load world", Q_NULLPTR));
        label_45->setText(QApplication::translate("MainWindow", "Organism blocks parameters modifiers", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem = table_organism_block_parameters->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("MainWindow", "Food Cost", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem1 = table_organism_block_parameters->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("MainWindow", "Life Point", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem2 = table_organism_block_parameters->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("MainWindow", "Chance Weight", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem3 = table_organism_block_parameters->verticalHeaderItem(0);
        ___qtablewidgetitem3->setText(QApplication::translate("MainWindow", "Mouth Cell", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem4 = table_organism_block_parameters->verticalHeaderItem(1);
        ___qtablewidgetitem4->setText(QApplication::translate("MainWindow", "Producer Cell", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem5 = table_organism_block_parameters->verticalHeaderItem(2);
        ___qtablewidgetitem5->setText(QApplication::translate("MainWindow", "Mover Cell", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem6 = table_organism_block_parameters->verticalHeaderItem(3);
        ___qtablewidgetitem6->setText(QApplication::translate("MainWindow", "Killer Cell", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem7 = table_organism_block_parameters->verticalHeaderItem(4);
        ___qtablewidgetitem7->setText(QApplication::translate("MainWindow", "Armor Cell", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem8 = table_organism_block_parameters->verticalHeaderItem(5);
        ___qtablewidgetitem8->setText(QApplication::translate("MainWindow", "Eye Cell", Q_NULLPTR));

        const bool __sortingEnabled = table_organism_block_parameters->isSortingEnabled();
        table_organism_block_parameters->setSortingEnabled(false);
        QTableWidgetItem *___qtablewidgetitem9 = table_organism_block_parameters->item(0, 0);
        ___qtablewidgetitem9->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem10 = table_organism_block_parameters->item(0, 1);
        ___qtablewidgetitem10->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem11 = table_organism_block_parameters->item(0, 2);
        ___qtablewidgetitem11->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem12 = table_organism_block_parameters->item(1, 0);
        ___qtablewidgetitem12->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem13 = table_organism_block_parameters->item(1, 1);
        ___qtablewidgetitem13->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem14 = table_organism_block_parameters->item(1, 2);
        ___qtablewidgetitem14->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem15 = table_organism_block_parameters->item(2, 0);
        ___qtablewidgetitem15->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem16 = table_organism_block_parameters->item(2, 1);
        ___qtablewidgetitem16->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem17 = table_organism_block_parameters->item(2, 2);
        ___qtablewidgetitem17->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem18 = table_organism_block_parameters->item(3, 0);
        ___qtablewidgetitem18->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem19 = table_organism_block_parameters->item(3, 1);
        ___qtablewidgetitem19->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem20 = table_organism_block_parameters->item(3, 2);
        ___qtablewidgetitem20->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem21 = table_organism_block_parameters->item(4, 0);
        ___qtablewidgetitem21->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem22 = table_organism_block_parameters->item(4, 1);
        ___qtablewidgetitem22->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem23 = table_organism_block_parameters->item(4, 2);
        ___qtablewidgetitem23->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem24 = table_organism_block_parameters->item(5, 0);
        ___qtablewidgetitem24->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem25 = table_organism_block_parameters->item(5, 1);
        ___qtablewidgetitem25->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem26 = table_organism_block_parameters->item(5, 2);
        ___qtablewidgetitem26->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        table_organism_block_parameters->setSortingEnabled(__sortingEnabled);

        Tabs->setTabText(Tabs->indexOf(world_controls_tab), QApplication::translate("MainWindow", "World Controls", Q_NULLPTR));
        label_4->setText(QApplication::translate("MainWindow", "Food production probability:", Q_NULLPTR));
        le_food_production_probability->setText(QApplication::translate("MainWindow", "0.5", Q_NULLPTR));
        label_15->setText(QApplication::translate("MainWindow", "Produce food every n tick:", Q_NULLPTR));
        le_produce_food_every_n_tick->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        label_5->setText(QApplication::translate("MainWindow", "llifespan multiplier:", Q_NULLPTR));
        le_lifespan_multiplier->setText(QApplication::translate("MainWindow", "100", Q_NULLPTR));
        label_7->setText(QApplication::translate("MainWindow", "look range:", Q_NULLPTR));
        le_look_range->setText(QApplication::translate("MainWindow", "50", Q_NULLPTR));
        label_9->setText(QApplication::translate("MainWindow", "auto produce n food:", Q_NULLPTR));
        le_auto_produce_n_food->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        label_44->setText(QApplication::translate("MainWindow", "auto produce food every n tick:", Q_NULLPTR));
        label_6->setText(QApplication::translate("MainWindow", "extra reproduction cost:", Q_NULLPTR));
        le_extra_reproduction_cost->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        cb_use_evolved_anatomy_mutation_rate->setText(QApplication::translate("MainWindow", "Use evolved anatomy mutation rate", Q_NULLPTR));
        lb_mutation_rate->setText(QApplication::translate("MainWindow", "Global anatomy mutation rate:", Q_NULLPTR));
        le_global_anatomy_mutation_rate->setText(QApplication::translate("MainWindow", "0.05", Q_NULLPTR));
        label_23->setText(QApplication::translate("MainWindow", "Anatomy mutation rate delimiter:", Q_NULLPTR));
        le_anatomy_mutation_rate_delimiter->setText(QApplication::translate("MainWindow", "0.5", Q_NULLPTR));
        cb_use_evolved_brain_mutation_rate->setText(QApplication::translate("MainWindow", "Use evolved brain mutation rate", Q_NULLPTR));
        label_20->setText(QApplication::translate("MainWindow", "Global brain mutation rate:", Q_NULLPTR));
        le_global_brain_mutation_rate->setText(QApplication::translate("MainWindow", "0.1", Q_NULLPTR));
        label_24->setText(QApplication::translate("MainWindow", "Brain mutation rate delimiter:", Q_NULLPTR));
        label_21->setText(QApplication::translate("MainWindow", "Killer damage amount:", Q_NULLPTR));
        le_killer_damage_amount->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        label_10->setText(QApplication::translate("MainWindow", "add cell:", Q_NULLPTR));
        le_add->setText(QApplication::translate("MainWindow", "33", Q_NULLPTR));
        label_11->setText(QApplication::translate("MainWindow", "change cell:", Q_NULLPTR));
        le_change->setText(QApplication::translate("MainWindow", "33", Q_NULLPTR));
        label_12->setText(QApplication::translate("MainWindow", "remove cell:", Q_NULLPTR));
        le_remove->setText(QApplication::translate("MainWindow", "33", Q_NULLPTR));
        cb_runtime_rotation_enabled->setText(QApplication::translate("MainWindow", "Runtime rotation enabled", Q_NULLPTR));
        cb_reproducing_rotation_enabled->setText(QApplication::translate("MainWindow", "Reproducing rotation enabled", Q_NULLPTR));
        cb_on_touch_kill->setText(QApplication::translate("MainWindow", "On touch kill", Q_NULLPTR));
        cb_movers_can_produce_food->setText(QApplication::translate("MainWindow", "Movers can produce food", Q_NULLPTR));
        cb_food_blocks_reproduction->setText(QApplication::translate("MainWindow", "Food blocks reproduction", Q_NULLPTR));
        cb_food_blocks_movement->setText(QApplication::translate("MainWindow", "Food blocks movement", Q_NULLPTR));
        label_17->setText(QApplication::translate("MainWindow", "Min reproducing distance: ", Q_NULLPTR));
        le_min_reproduction_distance->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        label_16->setText(QApplication::translate("MainWindow", "Max reproducing distance: ", Q_NULLPTR));
        le_max_reproduction_distance->setText(QApplication::translate("MainWindow", "3", Q_NULLPTR));
        cb_fix_reproduction_distance->setText(QApplication::translate("MainWindow", "Fix reproducing distance", Q_NULLPTR));
        cb_self_organism_blocks_block_sight->setText(QApplication::translate("MainWindow", "Organism's self cells block sight", Q_NULLPTR));
        cb_set_fixed_move_range->setText(QApplication::translate("MainWindow", "Set fixed move range", Q_NULLPTR));
        label_41->setText(QApplication::translate("MainWindow", "Min move range:", Q_NULLPTR));
        label_40->setText(QApplication::translate("MainWindow", "Max move range:", Q_NULLPTR));
        label_42->setText(QApplication::translate("MainWindow", "Move range delimiter", Q_NULLPTR));
        cb_failed_reproduction_eats_food->setText(QApplication::translate("MainWindow", "Failed reproduction eats food", Q_NULLPTR));
        cb_rotate_every_move_tick->setText(QApplication::translate("MainWindow", "Rotate every move tick", Q_NULLPTR));
        cb_multiply_food_production_prob->setText(QApplication::translate("MainWindow", "Multiply food production probability by number of producers", Q_NULLPTR));
        cb_simplified_food_production->setText(QApplication::translate("MainWindow", "Simplified food production", Q_NULLPTR));
        cb_stop_when_one_food_generated->setText(QApplication::translate("MainWindow", "Stop food production when one food is generated", Q_NULLPTR));
        cb_eat_then_produce->setText(QApplication::translate("MainWindow", "Eat first, then produce food.", Q_NULLPTR));
        Tabs->setTabText(Tabs->indexOf(evolution_controls_tab), QApplication::translate("MainWindow", "Evolution Controls", Q_NULLPTR));
        tb_open_statisctics->setText(QApplication::translate("MainWindow", "Open Statistics Window", Q_NULLPTR));
        cb_statistics_always_on_top->setText(QApplication::translate("MainWindow", "Always on top", Q_NULLPTR));
        Tabs->setTabText(Tabs->indexOf(statistics_tab), QApplication::translate("MainWindow", "Statistics", Q_NULLPTR));
        rb_single_thread_mode->setText(QApplication::translate("MainWindow", "Single thread CPU", Q_NULLPTR));
        rb_partial_multi_thread_mode->setText(QApplication::translate("MainWindow", "Partial multi-thread CPU", Q_NULLPTR));
        rb_multi_thread_mode->setText(QApplication::translate("MainWindow", "Multi-thread CPU", Q_NULLPTR));
        rb_cuda_mode->setText(QApplication::translate("MainWindow", "CUDA", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "Set number of CPU threads:", Q_NULLPTR));
        le_num_threads->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        label_19->setText(QApplication::translate("MainWindow", "Float number precision: ", Q_NULLPTR));
        le_float_number_precision->setText(QApplication::translate("MainWindow", "2", Q_NULLPTR));
        cb_disable_warnings->setText(QApplication::translate("MainWindow", "Disable warnings", Q_NULLPTR));
        cb_wait_for_engine_to_stop->setText(QApplication::translate("MainWindow", "Wait for engine to stop", Q_NULLPTR));
        label_25->setText(QApplication::translate("MainWindow", "Font size:", Q_NULLPTR));
        cb_simplified_rendering->setText(QApplication::translate("MainWindow", "Simplified rendering", Q_NULLPTR));
        label_46->setText(QApplication::translate("MainWindow", "Update info every n milliseconds:", Q_NULLPTR));
        cb_synchronise_info_with_window->setText(QApplication::translate("MainWindow", "Synhronise info updates with window updates", Q_NULLPTR));
        cb_use_nvidia_for_image_generation->setText(QApplication::translate("MainWindow", "Use NVIDIA GPU for image rendering", Q_NULLPTR));
        label_47->setText(QApplication::translate("MainWindow", "Menu minimum pixel height:", Q_NULLPTR));
        label_22->setText(QApplication::translate("MainWindow", "Perlin Settings:", Q_NULLPTR));
        label_26->setText(QApplication::translate("MainWindow", "Octaves: ", Q_NULLPTR));
        label_27->setText(QApplication::translate("MainWindow", "Persistence: ", Q_NULLPTR));
        label_28->setText(QApplication::translate("MainWindow", "Upper bound: ", Q_NULLPTR));
        label_29->setText(QApplication::translate("MainWindow", "Lower bound: ", Q_NULLPTR));
        label_30->setText(QApplication::translate("MainWindow", "x modifier: ", Q_NULLPTR));
        label_31->setText(QApplication::translate("MainWindow", "y modifier: ", Q_NULLPTR));
        Tabs->setTabText(Tabs->indexOf(settings_tab), QApplication::translate("MainWindow", "Settings", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WINDOWUI_H
