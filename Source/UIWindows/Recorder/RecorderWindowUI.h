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
#include <QtWidgets/QCheckBox>
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
    QTabWidget *teab;
    QWidget *full_video_image_creator;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *le_number_or_pixels_per_block;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_10;
    QLineEdit *le_kernel_size;
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
    QSpacerItem *verticalSpacer_4;
    QCheckBox *cb_use_relative_viewpoint;
    QCheckBox *cb_use_cuda;
    QCheckBox *cb_use_cuda_reconstructor;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_5;
    QLineEdit *le_viewpoint_x;
    QLabel *label_6;
    QLineEdit *le_viewpoint_y;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_7;
    QLineEdit *le_zoom;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_8;
    QLineEdit *le_image_width;
    QLabel *label_9;
    QLineEdit *le_image_height;
    QPushButton *b_set_from_camera;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *Recorder)
    {
        if (Recorder->objectName().isEmpty())
            Recorder->setObjectName("Recorder");
        Recorder->resize(948, 701);
        verticalLayout = new QVBoxLayout(Recorder);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        teab = new QTabWidget(Recorder);
        teab->setObjectName("teab");
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

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName("horizontalLayout_9");
        label_10 = new QLabel(full_video_image_creator);
        label_10->setObjectName("label_10");

        horizontalLayout_9->addWidget(label_10);

        le_kernel_size = new QLineEdit(full_video_image_creator);
        le_kernel_size->setObjectName("le_kernel_size");

        horizontalLayout_9->addWidget(le_kernel_size);


        verticalLayout_2->addLayout(horizontalLayout_9);

        b_create_image = new QPushButton(full_video_image_creator);
        b_create_image->setObjectName("b_create_image");

        verticalLayout_2->addWidget(b_create_image);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

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

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_4);

        cb_use_relative_viewpoint = new QCheckBox(full_video_image_creator);
        cb_use_relative_viewpoint->setObjectName("cb_use_relative_viewpoint");

        verticalLayout_2->addWidget(cb_use_relative_viewpoint);

        cb_use_cuda = new QCheckBox(full_video_image_creator);
        cb_use_cuda->setObjectName("cb_use_cuda");

        verticalLayout_2->addWidget(cb_use_cuda);

        cb_use_cuda_reconstructor = new QCheckBox(full_video_image_creator);
        cb_use_cuda_reconstructor->setObjectName("cb_use_cuda_reconstructor");

        verticalLayout_2->addWidget(cb_use_cuda_reconstructor);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName("horizontalLayout_6");
        label_5 = new QLabel(full_video_image_creator);
        label_5->setObjectName("label_5");

        horizontalLayout_6->addWidget(label_5);

        le_viewpoint_x = new QLineEdit(full_video_image_creator);
        le_viewpoint_x->setObjectName("le_viewpoint_x");

        horizontalLayout_6->addWidget(le_viewpoint_x);

        label_6 = new QLabel(full_video_image_creator);
        label_6->setObjectName("label_6");

        horizontalLayout_6->addWidget(label_6);

        le_viewpoint_y = new QLineEdit(full_video_image_creator);
        le_viewpoint_y->setObjectName("le_viewpoint_y");

        horizontalLayout_6->addWidget(le_viewpoint_y);


        verticalLayout_2->addLayout(horizontalLayout_6);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName("horizontalLayout_7");
        label_7 = new QLabel(full_video_image_creator);
        label_7->setObjectName("label_7");

        horizontalLayout_7->addWidget(label_7);

        le_zoom = new QLineEdit(full_video_image_creator);
        le_zoom->setObjectName("le_zoom");

        horizontalLayout_7->addWidget(le_zoom);


        verticalLayout_2->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName("horizontalLayout_8");
        label_8 = new QLabel(full_video_image_creator);
        label_8->setObjectName("label_8");

        horizontalLayout_8->addWidget(label_8);

        le_image_width = new QLineEdit(full_video_image_creator);
        le_image_width->setObjectName("le_image_width");

        horizontalLayout_8->addWidget(le_image_width);

        label_9 = new QLabel(full_video_image_creator);
        label_9->setObjectName("label_9");

        horizontalLayout_8->addWidget(label_9);

        le_image_height = new QLineEdit(full_video_image_creator);
        le_image_height->setObjectName("le_image_height");

        horizontalLayout_8->addWidget(le_image_height);


        verticalLayout_2->addLayout(horizontalLayout_8);

        b_set_from_camera = new QPushButton(full_video_image_creator);
        b_set_from_camera->setObjectName("b_set_from_camera");

        verticalLayout_2->addWidget(b_set_from_camera);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        teab->addTab(full_video_image_creator, QString());

        verticalLayout->addWidget(teab);


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
        QObject::connect(cb_use_relative_viewpoint, SIGNAL(toggled(bool)), Recorder, SLOT(cb_use_relative_viewpoint_slot(bool)));
        QObject::connect(le_viewpoint_x, SIGNAL(returnPressed()), Recorder, SLOT(le_viewpoint_x_slot()));
        QObject::connect(le_viewpoint_y, SIGNAL(returnPressed()), Recorder, SLOT(le_viewpoint_y_slot()));
        QObject::connect(le_zoom, SIGNAL(returnPressed()), Recorder, SLOT(le_zoom_slot()));
        QObject::connect(b_set_from_camera, SIGNAL(clicked()), Recorder, SLOT(b_set_from_camera_slot()));
        QObject::connect(le_image_width, SIGNAL(returnPressed()), Recorder, SLOT(le_image_width_slot()));
        QObject::connect(le_image_height, SIGNAL(returnPressed()), Recorder, SLOT(le_image_height_slot()));
        QObject::connect(cb_use_cuda, SIGNAL(toggled(bool)), Recorder, SLOT(cb_use_cuda_slot(bool)));
        QObject::connect(cb_use_cuda_reconstructor, SIGNAL(toggled(bool)), Recorder, SLOT(cb_use_cuda_reconstructor_slot(bool)));
        QObject::connect(le_kernel_size, SIGNAL(returnPressed()), Recorder, SLOT(le_kernel_size()));

        teab->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Recorder);
    } // setupUi

    void retranslateUi(QWidget *Recorder)
    {
        Recorder->setWindowTitle(QCoreApplication::translate("Recorder", "Recorder", nullptr));
        label->setText(QCoreApplication::translate("Recorder", "Number of pixels per world block", nullptr));
        label_10->setText(QCoreApplication::translate("Recorder", "Kernel size: ", nullptr));
        b_create_image->setText(QCoreApplication::translate("Recorder", "Create image", nullptr));
        label_2->setText(QCoreApplication::translate("Recorder", "Grid tbuffer size", nullptr));
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
        cb_use_relative_viewpoint->setText(QCoreApplication::translate("Recorder", "Use relative viewpoint", nullptr));
        cb_use_cuda->setText(QCoreApplication::translate("Recorder", "Use cuda", nullptr));
        cb_use_cuda_reconstructor->setText(QCoreApplication::translate("Recorder", "Use cuda reconstructor", nullptr));
        label_5->setText(QCoreApplication::translate("Recorder", "Viewpoint x ", nullptr));
        label_6->setText(QCoreApplication::translate("Recorder", "Viewpoint y ", nullptr));
        label_7->setText(QCoreApplication::translate("Recorder", "Zoom ", nullptr));
        label_8->setText(QCoreApplication::translate("Recorder", "Image width", nullptr));
        label_9->setText(QCoreApplication::translate("Recorder", "Image height ", nullptr));
        b_set_from_camera->setText(QCoreApplication::translate("Recorder", "Set from camera", nullptr));
        teab->setTabText(teab->indexOf(full_video_image_creator), QCoreApplication::translate("Recorder", "Full grid video/Image creator", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Recorder: public Ui_Recorder {};
} // namespace Ui

QT_END_NAMESPACE

#endif // RECORDERWINDOWUI_H
