/********************************************************************************
** Form generated from reading UI file 'benchmark.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef BENCHMARKUI_H
#define BENCHMARKUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Benchmark
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *main_tab;
    QVBoxLayout *verticalLayout_2;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_4;
    QSpacerItem *verticalSpacer_2;
    QPushButton *pushButton;
    QSpacerItem *verticalSpacer_4;
    QPushButton *pushButton_7;
    QPushButton *pushButton_9;
    QPushButton *pushButton_11;
    QPushButton *pushButton_12;
    QPushButton *pushButton_10;
    QPushButton *pushButton_8;
    QPushButton *pushButton_6;
    QPushButton *pushButton_5;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QPushButton *pushButton_13;
    QSpacerItem *verticalSpacer_3;
    QPushButton *pushButton_2;
    QSpacerItem *verticalSpacer;
    QVBoxLayout *verticalLayout_5;
    QTextEdit *textEdit;
    QWidget *settings_tab;

    void setupUi(QWidget *Benchmark)
    {
        if (Benchmark->objectName().isEmpty())
            Benchmark->setObjectName(QString::fromUtf8("Benchmark"));
        Benchmark->resize(1031, 618);
        verticalLayout = new QVBoxLayout(Benchmark);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Benchmark);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        main_tab = new QWidget();
        main_tab->setObjectName(QString::fromUtf8("main_tab"));
        verticalLayout_2 = new QVBoxLayout(main_tab);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        scrollArea = new QScrollArea(main_tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1025, 585));
        verticalLayout_3 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, -1, 0, 0);
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_2);

        pushButton = new QPushButton(scrollAreaWidgetContents);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        verticalLayout_4->addWidget(pushButton);

        verticalSpacer_4 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_4);

        pushButton_7 = new QPushButton(scrollAreaWidgetContents);
        pushButton_7->setObjectName(QString::fromUtf8("pushButton_7"));

        verticalLayout_4->addWidget(pushButton_7);

        pushButton_9 = new QPushButton(scrollAreaWidgetContents);
        pushButton_9->setObjectName(QString::fromUtf8("pushButton_9"));

        verticalLayout_4->addWidget(pushButton_9);

        pushButton_11 = new QPushButton(scrollAreaWidgetContents);
        pushButton_11->setObjectName(QString::fromUtf8("pushButton_11"));

        verticalLayout_4->addWidget(pushButton_11);

        pushButton_12 = new QPushButton(scrollAreaWidgetContents);
        pushButton_12->setObjectName(QString::fromUtf8("pushButton_12"));

        verticalLayout_4->addWidget(pushButton_12);

        pushButton_10 = new QPushButton(scrollAreaWidgetContents);
        pushButton_10->setObjectName(QString::fromUtf8("pushButton_10"));

        verticalLayout_4->addWidget(pushButton_10);

        pushButton_8 = new QPushButton(scrollAreaWidgetContents);
        pushButton_8->setObjectName(QString::fromUtf8("pushButton_8"));

        verticalLayout_4->addWidget(pushButton_8);

        pushButton_6 = new QPushButton(scrollAreaWidgetContents);
        pushButton_6->setObjectName(QString::fromUtf8("pushButton_6"));

        verticalLayout_4->addWidget(pushButton_6);

        pushButton_5 = new QPushButton(scrollAreaWidgetContents);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));

        verticalLayout_4->addWidget(pushButton_5);

        pushButton_3 = new QPushButton(scrollAreaWidgetContents);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));

        verticalLayout_4->addWidget(pushButton_3);

        pushButton_4 = new QPushButton(scrollAreaWidgetContents);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));

        verticalLayout_4->addWidget(pushButton_4);

        pushButton_13 = new QPushButton(scrollAreaWidgetContents);
        pushButton_13->setObjectName(QString::fromUtf8("pushButton_13"));

        verticalLayout_4->addWidget(pushButton_13);

        verticalSpacer_3 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_4->addItem(verticalSpacer_3);

        pushButton_2 = new QPushButton(scrollAreaWidgetContents);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        verticalLayout_4->addWidget(pushButton_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        textEdit = new QTextEdit(scrollAreaWidgetContents);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));
        textEdit->setReadOnly(true);

        verticalLayout_5->addWidget(textEdit);


        horizontalLayout->addLayout(verticalLayout_5);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 1);

        verticalLayout_3->addLayout(horizontalLayout);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_2->addWidget(scrollArea);

        tabWidget->addTab(main_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName(QString::fromUtf8("settings_tab"));
        tabWidget->addTab(settings_tab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Benchmark);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Benchmark);
    } // setupUi

    void retranslateUi(QWidget *Benchmark)
    {
        Benchmark->setWindowTitle(QApplication::translate("Benchmark", "Form", nullptr));
        pushButton->setText(QApplication::translate("Benchmark", "Full benchmark", nullptr));
        pushButton_7->setText(QApplication::translate("Benchmark", "Benchmark produce food", nullptr));
        pushButton_9->setText(QApplication::translate("Benchmark", "Benchmark eat food", nullptr));
        pushButton_11->setText(QApplication::translate("Benchmark", "Benchmark apply damage", nullptr));
        pushButton_12->setText(QApplication::translate("Benchmark", "Benchmark tick lifetime", nullptr));
        pushButton_10->setText(QApplication::translate("Benchmark", "Benchmark erase organisms", nullptr));
        pushButton_8->setText(QApplication::translate("Benchmark", "Benchmark reserve organisms", nullptr));
        pushButton_6->setText(QApplication::translate("Benchmark", "Benchmark get observations", nullptr));
        pushButton_5->setText(QApplication::translate("Benchmark", "Benchmark think decision", nullptr));
        pushButton_3->setText(QApplication::translate("Benchmark", "Benchmark rotate organism", nullptr));
        pushButton_4->setText(QApplication::translate("Benchmark", "Benchmark move organism", nullptr));
        pushButton_13->setText(QApplication::translate("Benchmark", "Benchmark try make child", nullptr));
        pushButton_2->setText(QApplication::translate("Benchmark", "Stop benchmark", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(main_tab), QApplication::translate("Benchmark", "Benchmark", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(settings_tab), QApplication::translate("Benchmark", "Benchmark Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Benchmark: public Ui_Benchmark {};
} // namespace Ui

QT_END_NAMESPACE

#endif // BENCHMARKUI_H
