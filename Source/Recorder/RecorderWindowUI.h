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
    QWidget *full_image_creator;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *le_number_or_pixels_per_block;
    QPushButton *b_create_image;
    QSpacerItem *verticalSpacer_2;
    QWidget *tab_2;

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
        full_image_creator = new QWidget();
        full_image_creator->setObjectName(QString::fromUtf8("full_image_creator"));
        verticalLayout_2 = new QVBoxLayout(full_image_creator);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(full_image_creator);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        le_number_or_pixels_per_block = new QLineEdit(full_image_creator);
        le_number_or_pixels_per_block->setObjectName(QString::fromUtf8("le_number_or_pixels_per_block"));

        horizontalLayout->addWidget(le_number_or_pixels_per_block);


        verticalLayout_2->addLayout(horizontalLayout);

        b_create_image = new QPushButton(full_image_creator);
        b_create_image->setObjectName(QString::fromUtf8("b_create_image"));

        verticalLayout_2->addWidget(b_create_image);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);

        tabWidget->addTab(full_image_creator, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        tabWidget->addTab(tab_2, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Recorder);
        QObject::connect(le_number_or_pixels_per_block, SIGNAL(returnPressed()), Recorder, SLOT(le_number_of_pixels_per_block_slot()));
        QObject::connect(b_create_image, SIGNAL(clicked()), Recorder, SLOT(b_create_image_slot()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Recorder);
    } // setupUi

    void retranslateUi(QWidget *Recorder)
    {
        Recorder->setWindowTitle(QApplication::translate("Recorder", "Recorder", nullptr));
        label->setText(QApplication::translate("Recorder", "Number of pixels per world block", nullptr));
        b_create_image->setText(QApplication::translate("Recorder", "Create image", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(full_image_creator), QApplication::translate("Recorder", "Full Image creator", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("Recorder", "Tab 2", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Recorder: public Ui_Recorder {};
} // namespace Ui

QT_END_NAMESPACE

#endif // RECORDERWINDOWUI_H
