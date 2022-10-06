/********************************************************************************
** Form generated from reading UI file 'recorder.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef RECORDERWINDOWUI_H
#define RECORDERWINDOWUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Recorder
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *full_video_image_creator;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *le_number_or_pixels_per_block;
    QPushButton *b_create_image;
    QSpacerItem *verticalSpacer_3;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QLineEdit *le_first_grid_buffer_size;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_3;
    QLineEdit *le_log_every_n_tick;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_4;
    QLineEdit *le_video_fps;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *b_new_recording;
    QPushButton *b_start_recording;
    QPushButton *b_pause_recording;
    QPushButton *b_stop_recording;
    QLabel *lb_recording_information;
    QPushButton *b_load_intermediate_data_location;
    QPushButton *b_compile_intermediate_data_into_video;
    QPushButton *b_clear_intermediate_data;
    QPushButton *b_delete_all_intermediate_data_from_disk;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *Recorder)
    {
        if (Recorder->objectName().isEmpty())
            Recorder->setObjectName("Recorder");
        Recorder->resize(900, 674);
        verticalLayout = new QVBoxLayout(Recorder);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Recorder);
        tabWidget->setObjectName("tabWidget");
        full_video_image_creator = new QWidget();
        full_video_image_creator->setObjectName("full_video_image_creator");
        verticalLayout_2 = new QVBoxLayout(full_video_image_creator);
        verticalLayout_2->setObjectName("verticalLayout_2");
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        label = new QLabel(full_video_image_creator);
        label->setObjectName("label");

        horizontalLayout->addWidget(label);

        le_number_or_pixels_per_block = new QLineEdit(full_video_image_creator);
        le_number_or_pixels_per_block->setObjectName("le_number_or_pixels_per_block");

        horizontalLayout->addWidget(le_number_or_pixels_per_block);


        verticalLayout_2->addLayout(horizontalLayout);

        b_create_image = new QPushButton(full_video_image_creator);
        b_create_image->setObjectName("b_create_image");

        verticalLayout_2->addWidget(b_create_image);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        label_2 = new QLabel(full_video_image_creator);
        label_2->setObjectName("label_2");

        horizontalLayout_2->addWidget(label_2);

        le_first_grid_buffer_size = new QLineEdit(full_video_image_creator);
        le_first_grid_buffer_size->setObjectName("le_first_grid_buffer_size");

        horizontalLayout_2->addWidget(le_first_grid_buffer_size);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        label_3 = new QLabel(full_video_image_creator);
        label_3->setObjectName("label_3");

        horizontalLayout_4->addWidget(label_3);

        le_log_every_n_tick = new QLineEdit(full_video_image_creator);
        le_log_every_n_tick->setObjectName("le_log_every_n_tick");

        horizontalLayout_4->addWidget(le_log_every_n_tick);


        verticalLayout_2->addLayout(horizontalLayout_4);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        label_4 = new QLabel(full_video_image_creator);
        label_4->setObjectName("label_4");

        horizontalLayout_5->addWidget(label_4);

        le_video_fps = new QLineEdit(full_video_image_creator);
        le_video_fps->setObjectName("le_video_fps");

        horizontalLayout_5->addWidget(le_video_fps);


        verticalLayout_2->addLayout(horizontalLayout_5);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        b_new_recording = new QPushButton(full_video_image_creator);
        b_new_recording->setObjectName("b_new_recording");

        horizontalLayout_3->addWidget(b_new_recording);

        b_start_recording = new QPushButton(full_video_image_creator);
        b_start_recording->setObjectName("b_start_recording");

        horizontalLayout_3->addWidget(b_start_recording);

        b_pause_recording = new QPushButton(full_video_image_creator);
        b_pause_recording->setObjectName("b_pause_recording");

        horizontalLayout_3->addWidget(b_pause_recording);

        b_stop_recording = new QPushButton(full_video_image_creator);
        b_stop_recording->setObjectName("b_stop_recording");

        horizontalLayout_3->addWidget(b_stop_recording);


        verticalLayout_2->addLayout(horizontalLayout_3);

        lb_recording_information = new QLabel(full_video_image_creator);
        lb_recording_information->setObjectName("lb_recording_information");
        lb_recording_information->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(lb_recording_information);

        b_load_intermediate_data_location = new QPushButton(full_video_image_creator);
        b_load_intermediate_data_location->setObjectName("b_load_intermediate_data_location");

        verticalLayout_2->addWidget(b_load_intermediate_data_location);

        b_compile_intermediate_data_into_video = new QPushButton(full_video_image_creator);
        b_compile_intermediate_data_into_video->setObjectName("b_compile_intermediate_data_into_video");

        verticalLayout_2->addWidget(b_compile_intermediate_data_into_video);

        b_clear_intermediate_data = new QPushButton(full_video_image_creator);
        b_clear_intermediate_data->setObjectName("b_clear_intermediate_data");

        verticalLayout_2->addWidget(b_clear_intermediate_data);

        b_delete_all_intermediate_data_from_disk = new QPushButton(full_video_image_creator);
        b_delete_all_intermediate_data_from_disk->setObjectName("b_delete_all_intermediate_data_from_disk");

        verticalLayout_2->addWidget(b_delete_all_intermediate_data_from_disk);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        tabWidget->addTab(full_video_image_creator, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Recorder);
        QObject::connect(le_number_or_pixels_per_block, SIGNAL(returnPressed()), Recorder, SLOT(le_number_of_pixels_per_block_slot()));
        QObject::connect(b_create_image, SIGNAL(clicked()), Recorder, SLOT(b_create_image_slot()));
        QObject::connect(le_first_grid_buffer_size, SIGNAL(returnPressed()), Recorder, SLOT(le_first_grid_buffer_size_slot()));
        QObject::connect(b_start_recording, SIGNAL(clicked()), Recorder, SLOT(b_start_recording_slot()));
        QObject::connect(b_stop_recording, SIGNAL(clicked()), Recorder, SLOT(b_stop_recording_slot()));
        QObject::connect(b_load_intermediate_data_location, SIGNAL(clicked()), Recorder, SLOT(b_load_intermediate_data_location_slot()));
        QObject::connect(b_compile_intermediate_data_into_video, SIGNAL(clicked()), Recorder, SLOT(b_compile_intermediate_data_into_video_slot()));
        QObject::connect(b_clear_intermediate_data, SIGNAL(clicked()), Recorder, SLOT(b_clear_intermediate_data_slot()));
        QObject::connect(b_delete_all_intermediate_data_from_disk, SIGNAL(clicked()), Recorder, SLOT(b_delete_all_intermediate_data_from_disk_slot()));
        QObject::connect(b_new_recording, SIGNAL(clicked()), Recorder, SLOT(b_new_recording_slot()));
        QObject::connect(b_pause_recording, SIGNAL(clicked()), Recorder, SLOT(b_pause_recording_slot()));
        QObject::connect(le_log_every_n_tick, SIGNAL(returnPressed()), Recorder, SLOT(le_log_every_n_tick_slot()));
        QObject::connect(le_video_fps, SIGNAL(returnPressed()), Recorder, SLOT(le_video_fps_slot()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Recorder);
    } // setupUi

    void retranslateUi(QWidget *Recorder)
    {
        Recorder->setWindowTitle(QCoreApplication::translate("Recorder", "Recorder", nullptr));
        label->setText(QCoreApplication::translate("Recorder", "Number of pixels per world block", nullptr));
        b_create_image->setText(QCoreApplication::translate("Recorder", "Create image", nullptr));
        label_2->setText(QCoreApplication::translate("Recorder", "Grid buffer size", nullptr));
        label_3->setText(QCoreApplication::translate("Recorder", "Log every n tick", nullptr));
        label_4->setText(QCoreApplication::translate("Recorder", "Video output FPS", nullptr));
        b_new_recording->setText(QCoreApplication::translate("Recorder", "New recording", nullptr));
        b_start_recording->setText(QCoreApplication::translate("Recorder", "Start recording", nullptr));
        b_pause_recording->setText(QCoreApplication::translate("Recorder", "Pause recording", nullptr));
        b_stop_recording->setText(QCoreApplication::translate("Recorder", "Stop recording", nullptr));
        lb_recording_information->setText(QCoreApplication::translate("Recorder", "Status: Stopped ||| Recorded 0 ticks", nullptr));
        b_load_intermediate_data_location->setText(QCoreApplication::translate("Recorder", "Load intermediate data location", nullptr));
        b_compile_intermediate_data_into_video->setText(QCoreApplication::translate("Recorder", "Compile intermediate data into video", nullptr));
        b_clear_intermediate_data->setText(QCoreApplication::translate("Recorder", "Clear intermediate data", nullptr));
        b_delete_all_intermediate_data_from_disk->setText(QCoreApplication::translate("Recorder", "Delete all intermediate data from disk", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(full_video_image_creator), QCoreApplication::translate("Recorder", "Full grid video/Image creator", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Recorder: public Ui_Recorder {};
} // namespace Ui

QT_END_NAMESPACE

#endif // RECORDERWINDOWUI_H
