/********************************************************************************
** Form generated from reading UI file 'recorder.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
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
    QHBoxLayout *horizontalLayout_3;
    QPushButton *b_start_recording;
    QPushButton *b_stop_recording;
    QLabel *lb_recording_information;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_3;
    QLineEdit *le_second_grid_buffer_size;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *b_load_intermediate_data_location;
    QPushButton *b_compile_intermediate_data_into_video;
    QPushButton *b_clear_intermediate_data;
    QPushButton *b_delete_all_intermediate_data_from_disk;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *Recorder)
    {
        if (Recorder->objectName().isEmpty())
            Recorder->setObjectName(QString::fromUtf8("Recorder"));
        Recorder->resize(900, 674);
        verticalLayout = new QVBoxLayout(Recorder);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Recorder);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        full_video_image_creator = new QWidget();
        full_video_image_creator->setObjectName(QString::fromUtf8("full_video_image_creator"));
        verticalLayout_2 = new QVBoxLayout(full_video_image_creator);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(full_video_image_creator);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        le_number_or_pixels_per_block = new QLineEdit(full_video_image_creator);
        le_number_or_pixels_per_block->setObjectName(QString::fromUtf8("le_number_or_pixels_per_block"));

        horizontalLayout->addWidget(le_number_or_pixels_per_block);


        verticalLayout_2->addLayout(horizontalLayout);

        b_create_image = new QPushButton(full_video_image_creator);
        b_create_image->setObjectName(QString::fromUtf8("b_create_image"));

        verticalLayout_2->addWidget(b_create_image);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_2->addItem(verticalSpacer_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_2 = new QLabel(full_video_image_creator);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_2->addWidget(label_2);

        le_first_grid_buffer_size = new QLineEdit(full_video_image_creator);
        le_first_grid_buffer_size->setObjectName(QString::fromUtf8("le_first_grid_buffer_size"));

        horizontalLayout_2->addWidget(le_first_grid_buffer_size);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        b_start_recording = new QPushButton(full_video_image_creator);
        b_start_recording->setObjectName(QString::fromUtf8("b_start_recording"));

        horizontalLayout_3->addWidget(b_start_recording);

        b_stop_recording = new QPushButton(full_video_image_creator);
        b_stop_recording->setObjectName(QString::fromUtf8("b_stop_recording"));

        horizontalLayout_3->addWidget(b_stop_recording);


        verticalLayout_2->addLayout(horizontalLayout_3);

        lb_recording_information = new QLabel(full_video_image_creator);
        lb_recording_information->setObjectName(QString::fromUtf8("lb_recording_information"));
        lb_recording_information->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(lb_recording_information);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_3 = new QLabel(full_video_image_creator);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_5->addWidget(label_3);

        le_second_grid_buffer_size = new QLineEdit(full_video_image_creator);
        le_second_grid_buffer_size->setObjectName(QString::fromUtf8("le_second_grid_buffer_size"));

        horizontalLayout_5->addWidget(le_second_grid_buffer_size);


        verticalLayout_2->addLayout(horizontalLayout_5);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        b_load_intermediate_data_location = new QPushButton(full_video_image_creator);
        b_load_intermediate_data_location->setObjectName(QString::fromUtf8("b_load_intermediate_data_location"));

        horizontalLayout_4->addWidget(b_load_intermediate_data_location);


        verticalLayout_2->addLayout(horizontalLayout_4);

        b_compile_intermediate_data_into_video = new QPushButton(full_video_image_creator);
        b_compile_intermediate_data_into_video->setObjectName(QString::fromUtf8("b_compile_intermediate_data_into_video"));

        verticalLayout_2->addWidget(b_compile_intermediate_data_into_video);

        b_clear_intermediate_data = new QPushButton(full_video_image_creator);
        b_clear_intermediate_data->setObjectName(QString::fromUtf8("b_clear_intermediate_data"));

        verticalLayout_2->addWidget(b_clear_intermediate_data);

        b_delete_all_intermediate_data_from_disk = new QPushButton(full_video_image_creator);
        b_delete_all_intermediate_data_from_disk->setObjectName(QString::fromUtf8("b_delete_all_intermediate_data_from_disk"));

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
        QObject::connect(le_second_grid_buffer_size, SIGNAL(returnPressed()), Recorder, SLOT(le_second_grid_buffer_size_slot()));
        QObject::connect(b_compile_intermediate_data_into_video, SIGNAL(clicked()), Recorder, SLOT(b_compile_intermediate_data_into_video_slot()));
        QObject::connect(b_clear_intermediate_data, SIGNAL(clicked()), Recorder, SLOT(b_clear_intermediate_data_slot()));
        QObject::connect(b_delete_all_intermediate_data_from_disk, SIGNAL(clicked()), Recorder, SLOT(b_delete_all_intermediate_data_from_disk_slot()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Recorder);
    } // setupUi

    void retranslateUi(QWidget *Recorder)
    {
        Recorder->setWindowTitle(QApplication::translate("Recorder", "Recorder", nullptr));
        label->setText(QApplication::translate("Recorder", "Number of pixels per world block", nullptr));
        b_create_image->setText(QApplication::translate("Recorder", "Create image", nullptr));
        label_2->setText(QApplication::translate("Recorder", "First grid buffer size", nullptr));
        b_start_recording->setText(QApplication::translate("Recorder", "Start recording", nullptr));
        b_stop_recording->setText(QApplication::translate("Recorder", "Stop recording", nullptr));
        lb_recording_information->setText(QApplication::translate("Recorder", "Status: Recording ||| Recorded n ticks ||| Total memory consumption n", nullptr));
        label_3->setText(QApplication::translate("Recorder", "Second grid buffer size", nullptr));
        b_load_intermediate_data_location->setText(QApplication::translate("Recorder", "Load intermediate data location", nullptr));
        b_compile_intermediate_data_into_video->setText(QApplication::translate("Recorder", "Compile intermediate data into video", nullptr));
        b_clear_intermediate_data->setText(QApplication::translate("Recorder", "Clear intermediate data", nullptr));
        b_delete_all_intermediate_data_from_disk->setText(QApplication::translate("Recorder", "Delete all intermediate data from disk", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(full_video_image_creator), QApplication::translate("Recorder", "Full grid video/Image creator", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Recorder: public Ui_Recorder {};
} // namespace Ui

QT_END_NAMESPACE

#endif // RECORDERWINDOWUI_H
