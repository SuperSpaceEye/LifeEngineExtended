/********************************************************************************
** Form generated from reading UI file 'editor.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef EDITORUI_H
#define EDITORUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Editor
{
public:
    QHBoxLayout *horizontalLayout;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QHBoxLayout *horizontalLayout_2;
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QStackedWidget *stackedWidget;
    QWidget *anatomy_edit_page;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QGraphicsView *editor_graphicsView;
    QVBoxLayout *verticalLayout_5;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout_5;
    QRadioButton *rb_mouth;
    QRadioButton *rb_producer;
    QRadioButton *rb_mover;
    QHBoxLayout *horizontalLayout_7;
    QRadioButton *rb_killer;
    QRadioButton *rb_armor;
    QRadioButton *rb_eye;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_7;
    QComboBox *cmb_block_rotation;
    QLabel *label_cell_count;
    QLabel *label_occ_count;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_2;
    QLineEdit *le_move_range;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_3;
    QLineEdit *le_anatomy_mutation_rate;
    QHBoxLayout *horizontalLayout_15;
    QLabel *label_6;
    QLineEdit *le_brain_mutation_rate;
    QHBoxLayout *horizontalLayout_13;
    QLabel *label_8;
    QComboBox *cmb_organism_rotation;
    QSpacerItem *verticalSpacer;
    QPushButton *b_reset_editing_view;
    QPushButton *b_reset_organism;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_4;
    QLineEdit *le_grid_width;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_5;
    QLineEdit *le_grid_height;
    QPushButton *b_resize_editing_grid;
    QSpacerItem *verticalSpacer_3;
    QWidget *brain_edit_page;
    QVBoxLayout *verticalLayout_6;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_14;
    QSpacerItem *horizontalSpacer_2;
    QVBoxLayout *brain_vertical_layout;
    QSpacerItem *horizontalSpacer;
    QWidget *page_2;
    QVBoxLayout *verticalLayout_10;
    QPlainTextEdit *te_occ_edit_window;
    QHBoxLayout *horizontalLayout_16;
    QPushButton *b_tcompile_occ;
    QCheckBox *cb_short_instructions;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_6;
    QRadioButton *rb_place_organism;
    QRadioButton *rb_chose_organism;
    QRadioButton *rb_null_button;
    QPushButton *b_save_organism;
    QPushButton *b_load_organism;
    QHBoxLayout *horizontalLayout_4;
    QRadioButton *rb_edit_anatomy;
    QRadioButton *rb_edit_brain;
    QRadioButton *rb_edit_occ;
    QButtonGroup *edit_group;
    QButtonGroup *organism_action_group;
    QButtonGroup *cells_group;

    void setupUi(QWidget *Editor)
    {
        if (Editor->objectName().isEmpty())
            Editor->setObjectName(QString::fromUtf8("Editor"));
        Editor->resize(940, 554);
        horizontalLayout = new QHBoxLayout(Editor);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(Editor);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 938, 552));
        horizontalLayout_2 = new QHBoxLayout(scrollAreaWidgetContents);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        widget = new QWidget(scrollAreaWidgetContents);
        widget->setObjectName(QString::fromUtf8("widget"));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(6, 6, 6, 6);
        stackedWidget = new QStackedWidget(widget);
        stackedWidget->setObjectName(QString::fromUtf8("stackedWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(stackedWidget->sizePolicy().hasHeightForWidth());
        stackedWidget->setSizePolicy(sizePolicy);
        anatomy_edit_page = new QWidget();
        anatomy_edit_page->setObjectName(QString::fromUtf8("anatomy_edit_page"));
        verticalLayout_3 = new QVBoxLayout(anatomy_edit_page);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        editor_graphicsView = new QGraphicsView(anatomy_edit_page);
        editor_graphicsView->setObjectName(QString::fromUtf8("editor_graphicsView"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(editor_graphicsView->sizePolicy().hasHeightForWidth());
        editor_graphicsView->setSizePolicy(sizePolicy1);
        editor_graphicsView->setMinimumSize(QSize(400, 400));
        editor_graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        editor_graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        editor_graphicsView->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);

        horizontalLayout_3->addWidget(editor_graphicsView);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalSpacer_2 = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_2);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        rb_mouth = new QRadioButton(anatomy_edit_page);
        cells_group = new QButtonGroup(Editor);
        cells_group->setObjectName(QString::fromUtf8("cells_group"));
        cells_group->addButton(rb_mouth);
        rb_mouth->setObjectName(QString::fromUtf8("rb_mouth"));
        rb_mouth->setChecked(true);

        horizontalLayout_5->addWidget(rb_mouth);

        rb_producer = new QRadioButton(anatomy_edit_page);
        cells_group->addButton(rb_producer);
        rb_producer->setObjectName(QString::fromUtf8("rb_producer"));

        horizontalLayout_5->addWidget(rb_producer);

        rb_mover = new QRadioButton(anatomy_edit_page);
        cells_group->addButton(rb_mover);
        rb_mover->setObjectName(QString::fromUtf8("rb_mover"));

        horizontalLayout_5->addWidget(rb_mover);


        verticalLayout_5->addLayout(horizontalLayout_5);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        rb_killer = new QRadioButton(anatomy_edit_page);
        cells_group->addButton(rb_killer);
        rb_killer->setObjectName(QString::fromUtf8("rb_killer"));

        horizontalLayout_7->addWidget(rb_killer);

        rb_armor = new QRadioButton(anatomy_edit_page);
        cells_group->addButton(rb_armor);
        rb_armor->setObjectName(QString::fromUtf8("rb_armor"));

        horizontalLayout_7->addWidget(rb_armor);

        rb_eye = new QRadioButton(anatomy_edit_page);
        cells_group->addButton(rb_eye);
        rb_eye->setObjectName(QString::fromUtf8("rb_eye"));

        horizontalLayout_7->addWidget(rb_eye);


        verticalLayout_5->addLayout(horizontalLayout_7);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_7 = new QLabel(anatomy_edit_page);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_12->addWidget(label_7);

        cmb_block_rotation = new QComboBox(anatomy_edit_page);
        cmb_block_rotation->addItem(QString());
        cmb_block_rotation->addItem(QString());
        cmb_block_rotation->addItem(QString());
        cmb_block_rotation->addItem(QString());
        cmb_block_rotation->setObjectName(QString::fromUtf8("cmb_block_rotation"));

        horizontalLayout_12->addWidget(cmb_block_rotation);


        verticalLayout_5->addLayout(horizontalLayout_12);

        label_cell_count = new QLabel(anatomy_edit_page);
        label_cell_count->setObjectName(QString::fromUtf8("label_cell_count"));

        verticalLayout_5->addWidget(label_cell_count);

        label_occ_count = new QLabel(anatomy_edit_page);
        label_occ_count->setObjectName(QString::fromUtf8("label_occ_count"));

        verticalLayout_5->addWidget(label_occ_count);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_2 = new QLabel(anatomy_edit_page);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        sizePolicy.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy);

        horizontalLayout_9->addWidget(label_2);

        le_move_range = new QLineEdit(anatomy_edit_page);
        le_move_range->setObjectName(QString::fromUtf8("le_move_range"));
        QSizePolicy sizePolicy2(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(le_move_range->sizePolicy().hasHeightForWidth());
        le_move_range->setSizePolicy(sizePolicy2);

        horizontalLayout_9->addWidget(le_move_range);


        verticalLayout_5->addLayout(horizontalLayout_9);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_3 = new QLabel(anatomy_edit_page);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        sizePolicy.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy);

        horizontalLayout_8->addWidget(label_3);

        le_anatomy_mutation_rate = new QLineEdit(anatomy_edit_page);
        le_anatomy_mutation_rate->setObjectName(QString::fromUtf8("le_anatomy_mutation_rate"));
        sizePolicy2.setHeightForWidth(le_anatomy_mutation_rate->sizePolicy().hasHeightForWidth());
        le_anatomy_mutation_rate->setSizePolicy(sizePolicy2);

        horizontalLayout_8->addWidget(le_anatomy_mutation_rate);


        verticalLayout_5->addLayout(horizontalLayout_8);

        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        label_6 = new QLabel(anatomy_edit_page);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_15->addWidget(label_6);

        le_brain_mutation_rate = new QLineEdit(anatomy_edit_page);
        le_brain_mutation_rate->setObjectName(QString::fromUtf8("le_brain_mutation_rate"));

        horizontalLayout_15->addWidget(le_brain_mutation_rate);


        verticalLayout_5->addLayout(horizontalLayout_15);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        label_8 = new QLabel(anatomy_edit_page);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_13->addWidget(label_8);

        cmb_organism_rotation = new QComboBox(anatomy_edit_page);
        cmb_organism_rotation->addItem(QString());
        cmb_organism_rotation->addItem(QString());
        cmb_organism_rotation->addItem(QString());
        cmb_organism_rotation->addItem(QString());
        cmb_organism_rotation->setObjectName(QString::fromUtf8("cmb_organism_rotation"));

        horizontalLayout_13->addWidget(cmb_organism_rotation);


        verticalLayout_5->addLayout(horizontalLayout_13);

        verticalSpacer = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer);

        b_reset_editing_view = new QPushButton(anatomy_edit_page);
        b_reset_editing_view->setObjectName(QString::fromUtf8("b_reset_editing_view"));

        verticalLayout_5->addWidget(b_reset_editing_view);

        b_reset_organism = new QPushButton(anatomy_edit_page);
        b_reset_organism->setObjectName(QString::fromUtf8("b_reset_organism"));

        verticalLayout_5->addWidget(b_reset_organism);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_4 = new QLabel(anatomy_edit_page);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_10->addWidget(label_4);

        le_grid_width = new QLineEdit(anatomy_edit_page);
        le_grid_width->setObjectName(QString::fromUtf8("le_grid_width"));

        horizontalLayout_10->addWidget(le_grid_width);


        verticalLayout_5->addLayout(horizontalLayout_10);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        label_5 = new QLabel(anatomy_edit_page);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_11->addWidget(label_5);

        le_grid_height = new QLineEdit(anatomy_edit_page);
        le_grid_height->setObjectName(QString::fromUtf8("le_grid_height"));

        horizontalLayout_11->addWidget(le_grid_height);


        verticalLayout_5->addLayout(horizontalLayout_11);

        b_resize_editing_grid = new QPushButton(anatomy_edit_page);
        b_resize_editing_grid->setObjectName(QString::fromUtf8("b_resize_editing_grid"));

        verticalLayout_5->addWidget(b_resize_editing_grid);

        verticalSpacer_3 = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_3);


        horizontalLayout_3->addLayout(verticalLayout_5);

        horizontalLayout_3->setStretch(1, 1);

        verticalLayout_2->addLayout(horizontalLayout_3);

        verticalLayout_2->setStretch(0, 1);

        verticalLayout_3->addLayout(verticalLayout_2);

        stackedWidget->addWidget(anatomy_edit_page);
        brain_edit_page = new QWidget();
        brain_edit_page->setObjectName(QString::fromUtf8("brain_edit_page"));
        verticalLayout_6 = new QVBoxLayout(brain_edit_page);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        widget_2 = new QWidget(brain_edit_page);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        horizontalLayout_14 = new QHBoxLayout(widget_2);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        horizontalSpacer_2 = new QSpacerItem(0, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_14->addItem(horizontalSpacer_2);

        brain_vertical_layout = new QVBoxLayout();
        brain_vertical_layout->setObjectName(QString::fromUtf8("brain_vertical_layout"));

        horizontalLayout_14->addLayout(brain_vertical_layout);

        horizontalSpacer = new QSpacerItem(0, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_14->addItem(horizontalSpacer);


        verticalLayout_6->addWidget(widget_2);

        stackedWidget->addWidget(brain_edit_page);
        page_2 = new QWidget();
        page_2->setObjectName(QString::fromUtf8("page_2"));
        verticalLayout_10 = new QVBoxLayout(page_2);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        te_occ_edit_window = new QPlainTextEdit(page_2);
        te_occ_edit_window->setObjectName(QString::fromUtf8("te_occ_edit_window"));

        verticalLayout_10->addWidget(te_occ_edit_window);

        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        b_tcompile_occ = new QPushButton(page_2);
        b_tcompile_occ->setObjectName(QString::fromUtf8("b_tcompile_occ"));

        horizontalLayout_16->addWidget(b_tcompile_occ);

        cb_short_instructions = new QCheckBox(page_2);
        cb_short_instructions->setObjectName(QString::fromUtf8("cb_short_instructions"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(cb_short_instructions->sizePolicy().hasHeightForWidth());
        cb_short_instructions->setSizePolicy(sizePolicy3);

        horizontalLayout_16->addWidget(cb_short_instructions);


        verticalLayout_10->addLayout(horizontalLayout_16);

        stackedWidget->addWidget(page_2);

        verticalLayout->addWidget(stackedWidget);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        rb_place_organism = new QRadioButton(widget);
        organism_action_group = new QButtonGroup(Editor);
        organism_action_group->setObjectName(QString::fromUtf8("organism_action_group"));
        organism_action_group->addButton(rb_place_organism);
        rb_place_organism->setObjectName(QString::fromUtf8("rb_place_organism"));
        rb_place_organism->setChecked(false);

        horizontalLayout_6->addWidget(rb_place_organism);

        rb_chose_organism = new QRadioButton(widget);
        organism_action_group->addButton(rb_chose_organism);
        rb_chose_organism->setObjectName(QString::fromUtf8("rb_chose_organism"));

        horizontalLayout_6->addWidget(rb_chose_organism);

        rb_null_button = new QRadioButton(widget);
        organism_action_group->addButton(rb_null_button);
        rb_null_button->setObjectName(QString::fromUtf8("rb_null_button"));
        rb_null_button->setEnabled(false);
        rb_null_button->setMaximumSize(QSize(0, 0));
        rb_null_button->setChecked(true);

        horizontalLayout_6->addWidget(rb_null_button);

        b_save_organism = new QPushButton(widget);
        b_save_organism->setObjectName(QString::fromUtf8("b_save_organism"));

        horizontalLayout_6->addWidget(b_save_organism);

        b_load_organism = new QPushButton(widget);
        b_load_organism->setObjectName(QString::fromUtf8("b_load_organism"));

        horizontalLayout_6->addWidget(b_load_organism);


        verticalLayout_4->addLayout(horizontalLayout_6);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        rb_edit_anatomy = new QRadioButton(widget);
        edit_group = new QButtonGroup(Editor);
        edit_group->setObjectName(QString::fromUtf8("edit_group"));
        edit_group->addButton(rb_edit_anatomy);
        rb_edit_anatomy->setObjectName(QString::fromUtf8("rb_edit_anatomy"));
        rb_edit_anatomy->setChecked(true);

        horizontalLayout_4->addWidget(rb_edit_anatomy);

        rb_edit_brain = new QRadioButton(widget);
        edit_group->addButton(rb_edit_brain);
        rb_edit_brain->setObjectName(QString::fromUtf8("rb_edit_brain"));

        horizontalLayout_4->addWidget(rb_edit_brain);

        rb_edit_occ = new QRadioButton(widget);
        edit_group->addButton(rb_edit_occ);
        rb_edit_occ->setObjectName(QString::fromUtf8("rb_edit_occ"));

        horizontalLayout_4->addWidget(rb_edit_occ);


        verticalLayout_4->addLayout(horizontalLayout_4);


        verticalLayout->addLayout(verticalLayout_4);


        horizontalLayout_2->addWidget(widget);

        scrollArea->setWidget(scrollAreaWidgetContents);

        horizontalLayout->addWidget(scrollArea);


        retranslateUi(Editor);
        QObject::connect(b_load_organism, SIGNAL(clicked()), Editor, SLOT(b_load_organism_slot()));
        QObject::connect(b_reset_editing_view, SIGNAL(clicked()), Editor, SLOT(b_reset_editing_view_slot()));
        QObject::connect(b_resize_editing_grid, SIGNAL(clicked()), Editor, SLOT(b_resize_editing_grid_slot()));
        QObject::connect(b_save_organism, SIGNAL(clicked()), Editor, SLOT(b_save_organism_slot()));
        QObject::connect(le_anatomy_mutation_rate, SIGNAL(returnPressed()), Editor, SLOT(le_anatomy_mutation_rate_slot()));
        QObject::connect(le_grid_height, SIGNAL(returnPressed()), Editor, SLOT(le_grid_height_slot()));
        QObject::connect(le_grid_width, SIGNAL(returnPressed()), Editor, SLOT(le_grid_width_slot()));
        QObject::connect(le_move_range, SIGNAL(returnPressed()), Editor, SLOT(le_move_range_slot()));
        QObject::connect(rb_armor, SIGNAL(clicked()), Editor, SLOT(rb_armor_slot()));
        QObject::connect(rb_chose_organism, SIGNAL(clicked()), Editor, SLOT(rb_choose_organism_slot()));
        QObject::connect(rb_edit_anatomy, SIGNAL(clicked()), Editor, SLOT(rb_edit_anatomy_slot()));
        QObject::connect(rb_edit_brain, SIGNAL(clicked()), Editor, SLOT(rb_edit_brain_slot()));
        QObject::connect(rb_eye, SIGNAL(clicked()), Editor, SLOT(rb_eye_slot()));
        QObject::connect(rb_killer, SIGNAL(clicked()), Editor, SLOT(rb_killer_slot()));
        QObject::connect(rb_mouth, SIGNAL(clicked()), Editor, SLOT(rb_mouth_slot()));
        QObject::connect(rb_mover, SIGNAL(clicked()), Editor, SLOT(rb_mover_slot()));
        QObject::connect(rb_place_organism, SIGNAL(clicked()), Editor, SLOT(rb_place_organism_slot()));
        QObject::connect(rb_producer, SIGNAL(clicked()), Editor, SLOT(rb_producer_slot()));
        QObject::connect(b_reset_organism, SIGNAL(clicked()), Editor, SLOT(b_reset_organism_slot()));
        QObject::connect(cmb_block_rotation, SIGNAL(currentTextChanged(QString)), Editor, SLOT(cmd_block_rotation_slot(QString)));
        QObject::connect(cmb_organism_rotation, SIGNAL(currentTextChanged(QString)), Editor, SLOT(cmd_organism_rotation_slot(QString)));
        QObject::connect(le_brain_mutation_rate, SIGNAL(returnPressed()), Editor, SLOT(le_brain_mutation_rate_slot()));
        QObject::connect(b_tcompile_occ, SIGNAL(clicked()), Editor, SLOT(b_compile_occ_slot()));
        QObject::connect(rb_edit_occ, SIGNAL(clicked()), Editor, SLOT(rb_edit_occ_slot()));
        QObject::connect(cb_short_instructions, SIGNAL(toggled(bool)), Editor, SLOT(cb_short_instructions_slot(bool)));

        stackedWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Editor);
    } // setupUi

    void retranslateUi(QWidget *Editor)
    {
        Editor->setWindowTitle(QApplication::translate("Editor", "Organism Editor", nullptr));
        rb_mouth->setText(QApplication::translate("Editor", "Mouth", nullptr));
        rb_producer->setText(QApplication::translate("Editor", "Producer", nullptr));
        rb_mover->setText(QApplication::translate("Editor", "Mover", nullptr));
        rb_killer->setText(QApplication::translate("Editor", "Killer", nullptr));
        rb_armor->setText(QApplication::translate("Editor", "Armor", nullptr));
        rb_eye->setText(QApplication::translate("Editor", "Eye", nullptr));
        label_7->setText(QApplication::translate("Editor", "Block Rotation:", nullptr));
        cmb_block_rotation->setItemText(0, QApplication::translate("Editor", "Up", nullptr));
        cmb_block_rotation->setItemText(1, QApplication::translate("Editor", "Left", nullptr));
        cmb_block_rotation->setItemText(2, QApplication::translate("Editor", "Down", nullptr));
        cmb_block_rotation->setItemText(3, QApplication::translate("Editor", "Right", nullptr));

        label_cell_count->setText(QApplication::translate("Editor", "Cell count:", nullptr));
        label_occ_count->setText(QApplication::translate("Editor", "OCC instruction count:", nullptr));
        label_2->setText(QApplication::translate("Editor", "Move range:", nullptr));
        label_3->setText(QApplication::translate("Editor", "Anatomy mutation rate:", nullptr));
        label_6->setText(QApplication::translate("Editor", "Brain mutation rate:", nullptr));
        label_8->setText(QApplication::translate("Editor", "Starting organism rotation:", nullptr));
        cmb_organism_rotation->setItemText(0, QApplication::translate("Editor", "Up", nullptr));
        cmb_organism_rotation->setItemText(1, QApplication::translate("Editor", "Left", nullptr));
        cmb_organism_rotation->setItemText(2, QApplication::translate("Editor", "Down", nullptr));
        cmb_organism_rotation->setItemText(3, QApplication::translate("Editor", "Right", nullptr));

        b_reset_editing_view->setText(QApplication::translate("Editor", "Reset View", nullptr));
        b_reset_organism->setText(QApplication::translate("Editor", "Reset organism", nullptr));
        label_4->setText(QApplication::translate("Editor", "Grid width:", nullptr));
        label_5->setText(QApplication::translate("Editor", "Grid height:", nullptr));
        b_resize_editing_grid->setText(QApplication::translate("Editor", "Resize editing grid", nullptr));
        b_tcompile_occ->setText(QApplication::translate("Editor", " Compile code", nullptr));
        cb_short_instructions->setText(QApplication::translate("Editor", "Short instructions", nullptr));
        rb_place_organism->setText(QApplication::translate("Editor", "Place Organism", nullptr));
        rb_chose_organism->setText(QApplication::translate("Editor", "Choose organism", nullptr));
        rb_null_button->setText(QString());
        b_save_organism->setText(QApplication::translate("Editor", "Save Organism", nullptr));
        b_load_organism->setText(QApplication::translate("Editor", "Load Organism", nullptr));
        rb_edit_anatomy->setText(QApplication::translate("Editor", "Edit Anatomy", nullptr));
        rb_edit_brain->setText(QApplication::translate("Editor", "Edit brain", nullptr));
        rb_edit_occ->setText(QApplication::translate("Editor", "Edit OCC", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Editor: public Ui_Editor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // EDITORUI_H
