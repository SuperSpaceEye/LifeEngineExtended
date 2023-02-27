/********************************************************************************
** Form generated from reading UI file 'info.ui'
**
** Created by: Qt User Interface Compiler version 6.4.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef INFOWINDOWUI_H
#define INFOWINDOWUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Info
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *OCC;
    QWidget *about_tab;
    QHBoxLayout *horizontalLayout;
    QTextEdit *textEdit;
    QWidget *evolution_controls_tab;
    QVBoxLayout *verticalLayout_2;
    QTextEdit *textEdit_2;
    QWidget *settings_tab;
    QVBoxLayout *verticalLayout_3;
    QTextEdit *textEdit_3;
    QWidget *recorder_tab;
    QVBoxLayout *verticalLayout_4;
    QTextEdit *textEdit_4;
    QWidget *world_events_tab;
    QVBoxLayout *verticalLayout_5;
    QTextEdit *textEdit_5;
    QWidget *tab;
    QVBoxLayout *verticalLayout_7;
    QTextEdit *textEdit_7;
    QWidget *custom_textures_tab;
    QVBoxLayout *verticalLayout_6;
    QTextEdit *textEdit_6;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_8;
    QTextEdit *textEdit_8;

    void setupUi(QWidget *Info)
    {
        if (Info->objectName().isEmpty())
            Info->setObjectName("Info");
        Info->resize(855, 739);
        verticalLayout = new QVBoxLayout(Info);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        OCC = new QTabWidget(Info);
        OCC->setObjectName("OCC");
        about_tab = new QWidget();
        about_tab->setObjectName("about_tab");
        horizontalLayout = new QHBoxLayout(about_tab);
        horizontalLayout->setObjectName("horizontalLayout");
        textEdit = new QTextEdit(about_tab);
        textEdit->setObjectName("textEdit");
        textEdit->setReadOnly(true);

        horizontalLayout->addWidget(textEdit);

        OCC->addTab(about_tab, QString());
        evolution_controls_tab = new QWidget();
        evolution_controls_tab->setObjectName("evolution_controls_tab");
        verticalLayout_2 = new QVBoxLayout(evolution_controls_tab);
        verticalLayout_2->setObjectName("verticalLayout_2");
        textEdit_2 = new QTextEdit(evolution_controls_tab);
        textEdit_2->setObjectName("textEdit_2");
        textEdit_2->setReadOnly(true);

        verticalLayout_2->addWidget(textEdit_2);

        OCC->addTab(evolution_controls_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName("settings_tab");
        verticalLayout_3 = new QVBoxLayout(settings_tab);
        verticalLayout_3->setObjectName("verticalLayout_3");
        textEdit_3 = new QTextEdit(settings_tab);
        textEdit_3->setObjectName("textEdit_3");
        textEdit_3->setReadOnly(true);

        verticalLayout_3->addWidget(textEdit_3);

        OCC->addTab(settings_tab, QString());
        recorder_tab = new QWidget();
        recorder_tab->setObjectName("recorder_tab");
        verticalLayout_4 = new QVBoxLayout(recorder_tab);
        verticalLayout_4->setObjectName("verticalLayout_4");
        textEdit_4 = new QTextEdit(recorder_tab);
        textEdit_4->setObjectName("textEdit_4");
        textEdit_4->setReadOnly(true);

        verticalLayout_4->addWidget(textEdit_4);

        OCC->addTab(recorder_tab, QString());
        world_events_tab = new QWidget();
        world_events_tab->setObjectName("world_events_tab");
        verticalLayout_5 = new QVBoxLayout(world_events_tab);
        verticalLayout_5->setObjectName("verticalLayout_5");
        textEdit_5 = new QTextEdit(world_events_tab);
        textEdit_5->setObjectName("textEdit_5");
        textEdit_5->setReadOnly(true);

        verticalLayout_5->addWidget(textEdit_5);

        OCC->addTab(world_events_tab, QString());
        tab = new QWidget();
        tab->setObjectName("tab");
        verticalLayout_7 = new QVBoxLayout(tab);
        verticalLayout_7->setObjectName("verticalLayout_7");
        textEdit_7 = new QTextEdit(tab);
        textEdit_7->setObjectName("textEdit_7");
        textEdit_7->setReadOnly(true);

        verticalLayout_7->addWidget(textEdit_7);

        OCC->addTab(tab, QString());
        custom_textures_tab = new QWidget();
        custom_textures_tab->setObjectName("custom_textures_tab");
        verticalLayout_6 = new QVBoxLayout(custom_textures_tab);
        verticalLayout_6->setObjectName("verticalLayout_6");
        textEdit_6 = new QTextEdit(custom_textures_tab);
        textEdit_6->setObjectName("textEdit_6");
        textEdit_6->setReadOnly(true);

        verticalLayout_6->addWidget(textEdit_6);

        OCC->addTab(custom_textures_tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName("tab_2");
        verticalLayout_8 = new QVBoxLayout(tab_2);
        verticalLayout_8->setObjectName("verticalLayout_8");
        textEdit_8 = new QTextEdit(tab_2);
        textEdit_8->setObjectName("textEdit_8");
        textEdit_8->setReadOnly(true);

        verticalLayout_8->addWidget(textEdit_8);

        OCC->addTab(tab_2, QString());

        verticalLayout->addWidget(OCC);


        retranslateUi(Info);

        OCC->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Info);
    } // setupUi

    void retranslateUi(QWidget *Info)
    {
        Info->setWindowTitle(QCoreApplication::translate("Info", "Info Window", nullptr));
        textEdit->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"lifeengine-extended\"></a><span style=\" font-size:xx-large; font-weight:600;\">L</span><span style=\" font-size:xx-large; font-weight:600;\">ifeEngine Extended</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Web Version: <a href=\"https://lifeengineextended.github.io/\"><span style=\" text-decoration: underline; color:#0000ff;\">https://lifeengineextended.github.io/</span></a></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-righ"
                        "t:0px; -qt-block-indent:0; text-indent:0px;\">Warning! The web version is made using qt6 webassembly compiler which autogenerates a lot of stuff i don't know how to debug or fix. So if you have any bugs, glitches or errors (in the web version), i will probably be unable to fix them</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Idea for this project is based on <a href=\"https://github.com/MaxRobinsonTheGreat/LifeEngine\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/MaxRobinsonTheGreat/LifeEngine</span></a>.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The program needs to be placed in path with only english letters.</p>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"important-information-\"></a><span style=\" font-size:"
                        "xx-large; font-weight:600;\">I</span><span style=\" font-size:xx-large; font-weight:600;\">mportant information:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You should press enter after inputing text in line edits or it will not register.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Setting -1 into max fps/sps/organisms line edits will disable the limit.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">To use keyboard actions you should click on simulation window first.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When"
                        " you use &quot;Choose organism&quot; mouse mode it will search for any organism in brush area and will stop when it finds one.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can use mouse actions in editor.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Keyboard actions do not work in editor.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"mouse-actions-\"></a><span style=\" font-size:xx-large; font-weight:600;\">M</span><span style=\" font-size:xx-large; font-weight:600;\">ouse actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0"
                        "; text-indent:0px;\">Hold left button to place/kill.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold right button to delete/kill</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold scroll button to move the view.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Scroll to zoom in/out.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"keyboard-button-actions-\"></a><span style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">eyboard button actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style"
                        "=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;W&quot; or &quot;Up&quot; - move view up.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;A&quot; or &quot;Left&quot; - move view left.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;S&quot; or &quot;Down&quot; - move view down.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;D&quot; or &quot;Right&quot; - move view right.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;Shift&quot; - hold to make movement using keys 2 times as fast.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-"
                        "bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;R&quot; - reset view.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;M&quot; - hide/show menu.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;Space bar&quot; - pause simulation,</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;.&quot; - pass one simulation tick.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;numpad +&quot; - zoom in.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;numpad -&quot; - zoom out.</li>\n"
"<li style=\""
                        "\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;1&quot; - switch mouse mode to place/delete food.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;2&quot; - switch mouse mode to kill organisms.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;3&quot; - switch mouse mode to place/delete walls.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;4&quot; - switch mouse mode to place editor organisms.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;5&quot; - switch mouse mode to choose organisms from simulation.</li></ul>\n"
"<h1 style=\" margin-top"
                        ":18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"known-bugs-\"></a><span style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">nown bugs:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Saving and loading will not work correctly unless your path contains only english letters.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Mouse movement tracking is imprecise.</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(about_tab), QCoreApplication::translate("Info", "About", nullptr));
        textEdit_2->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"evolution-controls\"></a><span style=\" font-size:xx-large; font-weight:600;\">E</span><span style=\" font-size:xx-large; font-weight:600;\">volution Controls</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">These are the explanations for some evolution options</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right"
                        ":0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Food production probability&quot;</span> - Probablilty of producing food per tick for each producing block</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Lifespan multiplier&quot;</span> - Multiplicator of the sum of &quot;Lifetime Weight&quot; of each block of an organism.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Anatomy mutation rate mutation step&quot;</span> - An amount by which a mutation rate of an organism can increase or decrease</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Anatomy mutation rate delimiter&quot;</s"
                        "pan> - A parameter which controls whatever organism's anatomy mutation rate will be biased to increase or decrease. If &gt;0.5 then the rate will increase, if &lt;0.5 then the rate will decrease, if == 0.5 then no bias.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Brain mutation rate mutation step&quot;</span> - The same as &quot;Anatomy mutation rate mutation step&quot;</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Brain mutation rate delimiter&quot;</span> - The same as &quot;Anatomy mutation rate delimiter&quot;</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Fix reproduction distance&quot;</span> - Will mak"
                        "e reproduction distance always equal to &quot;Min reproduction distance&quot; during reproduction.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Organism's self cells block sight&quot;</span> - If disabled, organism can see through itself. If enabled, the eye that points to the cell that belongs to itself will return &quot;Empty block&quot; as observation.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Set fixed move range&quot;</span> - Will force organisms to use &quot;Min move range&quot; and will make child move ranges equal to parent during reproduction.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Move r"
                        "ange delimiter&quot;</span> - The same as Anatomy mutation rate delimiter\342\200\235</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Food threshold&quot;</span> - Minimum amount of food needed in a world block before organisms can eat it.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Max food&quot;</span> - Maximum amount of food a world block can contain</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Continuous movement drag&quot;</span> - If continuous mode enabled, sets drag for organism's movement</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block"
                        "-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Failed reproduction eats food&quot;</span> - If disabled, then the food will be deduced from parent organism only after successful reproduction.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Rotate every move tick&quot;</span> - Will make organisms rotate every time they move. If disabled, then they will rotate only at the end of move range.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Simplified food production&quot;</span> - Will try to produce food for each space that can produce food.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Eat fi"
                        "rst, then produce food&quot;</span> - If disabled, will produce food first and only then eat.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Use new child position calculator&quot;</span> - New child position calculator calculating position of a child by first calculating the coordinates of edges of parent and child + distance. For example, if the chosen reproduction direction is up, then calculator will calculate the uppermost y cell coordinate of a parent, the bottom y cell coordinate of a child + distance. That way, the child organism will never appear inside a parent organism_index. The old child position calculator however calculates only the edge coordinates of a parent organism_index + distance, allowing child organisms to appear inside parent, with the side effect of organisms being unable to reproduce if the reproducing distance is less than (height or width)/2 (depend"
                        "ing on child organism_index rotation and chosen reproductive direction)</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Check if path is clear&quot;</span> - If enabled, will check for each cell of a child organism if there is an obstruction in the way (except for parents cells), like a wall or a food if &quot;Food blocks reproduction&quot; is enabled. If there is, then the parent organism_index will not reproduce. If disabled, the child will just check if there is a space for itself at the end coordinated, but it will introduce some behaviours such as child organisms hopping though walls if they are thin enough.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Recenter organism to imaginary position&quot;</span> - If enabled, will set &quot;center"
                        "&quot; of organism to mathematical center, even if there are no cell corresponding to center.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Use weighted brain&quot;</span> - Instead of discrete decision making, will add all observations to a movement vector.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;Organism cell parameters modifiers&quot; - modifiers for each cell of all organisms.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Life point&quot;</span> - The amount of life points this cell will give to organism</li>\n"
"<li style=\"\" sty"
                        "le=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Lifetime weight&quot;</span> - The amount of lifetime this cell will give to organism</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Chance Weight&quot;</span> - Controls how likely this cell will be picked during reproduction compared to others. If 0, the cell will never get picked. </li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(evolution_controls_tab), QCoreApplication::translate("Info", "Evolution Controls", nullptr));
        textEdit_3->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Settings_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">S</span><span style=\" font-size:xx-large; font-weight:600;\">ettings</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Float number precision\342\200\235</span> - A decorative parameter that control how many zeros after the decimal"
                        " point of floats will be displayed in labels</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Wait for engine to stop to render\342\200\235</span> - If enabled, will send an engine a signal to stop before rendering simulation grid.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Really stop render\342\200\235</span>- To render an image, the program first calculates what cells will be seen by user, and then it copies them to the secondary grid containing only cell type and rotation, from which the image is constructed. If disabled, will parse the whole grid when \342\200\234Stop render\342\200\235 button is pressed, which will allow to move and scale the view. If enabled, will not parse the grid or construct an image when \342\200\234St"
                        "op render\342\200\235 is pressed.</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(settings_tab), QCoreApplication::translate("Info", "Settings", nullptr));
        textEdit_4->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"recorder\"></a><span style=\" font-size:xx-large; font-weight:600;\">R</span><span style=\" font-size:xx-large; font-weight:600;\">ecorder</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Recorder can either create photos or videos of full grid. To create a photo just click on &quot;Create image&quot;. To make a video follow these steps:</p>\n"
"<ol style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\""
                        "\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">&quot;New recording&quot;</span></li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">&quot;Start recording&quot;</span></li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Record for however long you want.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">&quot;Pause recording&quot;</span></li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">&quot;Compile intermediate data into v"
                        "ideo&quot;</span></li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Wait until video is compiled. Finished video will be in the &quot;path to program dir/videos&quot;. The video will be named with the timestamp of the start of the recording</li></ol>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can compile several videos in parallel, but it is not recommended.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"various-setting-\"></a><span style=\" font-size:large; font-weight:600;\">V</span><span style=\" font-size:large; font-weight:600;\">arious setting.</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-l"
                        "eft:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Grid tbuffer size&quot;</span> - The size of a tbuffer. The program will first write grid states to the tbuffer, before writing all at once to the drive.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Log every n tick&quot;</span> - Will log every n tick.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Video output FPS&quot;</span> - FPS that will be set during video construction.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"various-buttons-\"></a><span style=\" font-size:large; font-weight:600;\">V</span><span style=\" font-size:large; fon"
                        "t-weight:600;\">arious buttons.</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;New recording&quot;</span> - Will create new folder in /temp/ with timestamp where recording will be stored.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Stop recording&quot;</span> - Will stop the recording, flushing data in the tbuffer to the disk.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Clear intermediate data&quot;</span> - Will stop the recording before freeing the tbuffer space.</li>\n"
"<li style=\"\" styl"
                        "e=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Delete all intermediate data from disk&quot;</span> - Will delete everything in the /temp/ folder.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Compile intermediate data into video&quot;</span> - The output will be in /videos/ folder. Compilation is done in two stages:</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">&quot;Load intermediate data location&quot;</span> - Choose the folder with the recording. </li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(recorder_tab), QCoreApplication::translate("Info", "Recorder", nullptr));
        textEdit_5->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"World_Events_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">W</span><span style=\" font-size:xx-large; font-weight:600;\">orld Events</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events are events that are executed after simulation tick.<br />Right now there is two types of events: \342\200\234Conditional\342\200\235, \342\200\234Change Value\342\200\235.<br />World events are divided into rows and nodes. First executed rows from the bott"
                        "om, then inside rows  nodes are executed left to right.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events have two settings - \342\200\234Update World Events every n ticks\342\200\235 and \342\200\234Collect Info every n ticks\342\200\235.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Update World Events every n ticks\342\200\235</span> - Although world events are pretty lightweight in regard to performance, they are not free. So I made this parameter to control when nodes are updated.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234C"
                        "ollect Info every n ticks\342\200\235</span> - The conditional node needs info to make decision.<br />If the value of this parameter is too large the conditional node will use an outdated data.<br />However, if the value is too small it will hurt the performance, as gathering data of simulation is not free.<br />Be aware, that because world events and statistics share the same location of info, when UI part sends signal to update data, it will also update data for the world events.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Every node has \342\200\234Update every n tick\342\200\235. It works the same as \342\200\234Update World Events every n ticks\342\200\235. When last execution time exceeds this parameter, the node will execute.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Each world events branch also has \342\200\234Repeat branc"
                        "h\342\200\235. If event branch reaches the end and parameter was toggled, the execution will begin from the start, else it will just stop.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events in \342\200\234World Events Editor\342\200\235 and simulation are separate, so changes in the editor will not affect world events in simulation, unless you apply them.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Conditional_Event_18\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">onditional Event</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When event branch reaches conditional node, it will continuously check if the statement is true. If it is, the execution of the next node w"
                        "ill begin, otherwise it will repeat the check.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Change_Value_21\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">hange Value</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Change value node allows for World Events to actually influence the simulation.<br />With this node you can change \342\200\234some\342\200\235 simulation and block parameters.<br />This node has several modes with how it can change selected value.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Linear\342"
                        "\200\235</span> - Will change the value to target across time.<br />The parameter \342\200\234Time horizon\342\200\235 controls for how long the value is changed.<br />During the execution of this mode, any changes to the value will not be applied.<br />If two Linear nodes started executing at the same time, and have the same time horizon, the one in higher branch will set final target value, otherwise the one finishing last will set the value.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Step\342\200\235</span> - Will change the value to target value upon reaching.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The modes below were added by omgdev. All nodes are executed upon reaching.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -"
                        "qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Increase by\342\200\235</span> - Will increase chosen variable by target amount.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Decrease by\342\200\235</span> - Will decrease chosen variable by target amount.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Multiply by\342\200\235</span> - Will multiply chosen variable by target amount.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Divide by\342\200"
                        "\235</span> - Will divide chosen variable by target amount.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Running_World_Events_37\"></a><span style=\" font-size:large; font-weight:600;\">R</span><span style=\" font-size:large; font-weight:600;\">unning World Events</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">After creating World Events click \342\200\234Apply events\342\200\235.<br />World events will not be applied if there are nodes without set value.<br />If applying was successful, the world events will start execution.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">While the world events are running, you can\342\200\231t change some values.<br />To change the values, pause the simulation or pause execution of world events in the "
                        "tab \342\200\234Current World Events Viewer\342\200\235 of world events window.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">If simulation resets, world events will also automatically reset and start from the beginning. If execution of world events is stopped or the simulation resets, the simulation settings will be set to the state they were before the execution of world events started, unless you stop them with \342\200\234Stop Events No Setting Reset\342\200\235.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can pause/resume execution of world events with buttons \342\200\234Pause events\342\200\235/\342\200\234Resume events\342\200\235. These buttons will not reset world events.<br />If world events are already applied and were stopped, you can use \342\200\234Start Events\342\200\235 button to re-enable already applied world events.<"
                        "/p></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(world_events_tab), QCoreApplication::translate("Info", "World Events", nullptr));
        textEdit_7->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"organism-construction-code\"></a><span style=\" font-size:xx-large; font-weight:600;\">O</span><span style=\" font-size:xx-large; font-weight:600;\">rganism Construction Code</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Organism Construction Code or OCC is a way to represent anatomy of organisms as a DNA like structure. The OCC is an array of different instructions, executed one after another. The anatomy is constructed on a grid, with &quot;cursor&quot; bein"
                        "g the main component. During the process of anatomy constructions, different elements are used:</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"construction-elements-\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">onstruction elements:</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;cursor&quot; - a point in space depending on a position of which organism blocks are placed.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;origin&quot; - a changeable position that a cursor can return to if appropriate command is executed.</li>\n"
"<li style=\""
                        "\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;rotation cursor&quot; - the base rotation of blocks that will be placed with.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;group&quot; - a sequence of random instruction with the size randomly chosen between [1, max_size].</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The OCC can be viewed and edited in an &quot;edit OCC&quot; option in editor window.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"rules-of-occ-edit-window-\"></a><span style=\" font-size:large; font-weight:600;\">R</span><span style=\" font-size:large; font-weight:600;\">ules of OCC edit window:</span></h3>\n"
"<ul style=\"margin-top: "
                        "0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">To write a sequence of instructions, each instruction must be a valid instruction.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Instruction must be finished with a &quot;;&quot; sign.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The spaces, indentation or new lines do not affect the execution of OCC.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can make comment with a &quot;/&quot; sign.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0"
                        "; text-indent:0px;\">After execution, OCC must create at least 1 organism block.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Instructions can be written in either full or shortened form.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"all-occ-mutations-\"></a><span style=\" font-size:large; font-weight:600;\">A</span><span style=\" font-size:large; font-weight:600;\">ll OCC mutations:</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Append random - appends a group to the end of OCC sequence.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:"
                        "0; text-indent:0px;\">Insert random - inserts a group to a random uniformly sampled position in OCC sequence.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Change random - first gets a group, then randomly chooses the position inside the OCC sequence, then overwrites the part of existing OCC with instructions from group until either a group is fully written, or OCC reaches end.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Delete random - will delete OCC instruction the size of group starting from random position, until either the number of instructions deleted = group size, or OCC reaches end.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Swap random - will choose random position, randomly decide the direction of movement (le"
                        "ft or right), randomly decide the distance instructions will be moved, randomly decide the distance from chosen position, then it will start to swap elements of group size, until either it successfully swapped the elements, or OCC ends.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"all-occ-instructions-\"></a><span style=\" font-size:large; font-weight:600;\">A</span><span style=\" font-size:large; font-weight:600;\">ll OCC instructions:</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Shift Instructions - Shifts cursor to direction, or if there is SetBlock instruction after it, sets the block above the cursor, not changing it's position.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-lef"
                        "t:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftUp or SU</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftUpLeft or SUL</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftLeft or SL</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftLeftDown or SLD</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftDown or SD</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftDownRight or SDR</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">S"
                        "hiftRight or SR</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftRightUp or SRU</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Apply rotation instructions - Sets the rotation of rotation cursor to the new rotation.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationUp or ARU</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationLeft or ARL</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationDown or ARD</li>\n"
"<li s"
                        "tyle=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationRight or ARR</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set rotation instructions - Sets the rotation of a block directly underneath the cursor to the new rotation. Will work only if there is already a block underneath the cursor.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationUp or SRTU</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationLeft or SRL</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0"
                        "; text-indent:0px;\">SetRotationDown or SRD</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationRight or SRR</li></ul>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ResetToOrigin or RTO - Resets the position of a cursor to the position of origin.</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetOrigin or SO - Sets the position of an origin to the position of a cursor.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set block instructions - - Sets the block on a grid to type.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; "
                        "margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockMouth or SBM</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockProducer or SBP</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockMover or SBMV</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockKiller or SBK</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockArmor or SBA</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockEye"
                        " or SBE</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set under block instructions - Sets the block on a grid to mouth directly underneath of the cursor. Is not affected by shift instructions</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockMouth or SUBM</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockProducer or SUBP</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockMover or SUBMV</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; "
                        "text-indent:0px;\">SetUnderBlockKiller or SURK</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockArmor or SUBA</li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockEye or SUBE </li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(tab), QCoreApplication::translate("Info", "OCC", nullptr));
        textEdit_6->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Custom_Textures_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">C</span><span style=\" font-size:xx-large; font-weight:600;\">ustom Textures</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Life Engine Extended allows you to set arbitrary images as textures for organism cells, food, walls and empty space.<br />For image to be usable by program, it needs to be renamed to one of these titles: <span style=\" font-weight:600;\">(\342\200\234empty\342\200"
                        "\235, \342\200\234mouth\342\200\235, \342\200\234producer\342\200\235, \342\200\234mover\342\200\235, \342\200\234killer\342\200\235, \342\200\234armor\342\200\235, \342\200\234eye\342\200\235, \342\200\234food\342\200\235, \342\200\234wall\342\200\235)</span> and be placed into <span style=\" font-weight:600;\">\342\200\234/textures/\342\200\235</span> folder.<br />These textures will be loaded upon program loading, or if you click <span style=\" font-weight:600;\">\342\200\234Update textures\342\200\235</span>.<br />If there is no image with type in <span style=\" font-weight:600;\">\342\200\234/textures/\342\200\235</span> folder, the program will revert to base texture.<br />The images can be of any size, but the program will stretch them to box shape.</p></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(custom_textures_tab), QCoreApplication::translate("Info", "Custom Textures", nullptr));
        textEdit_8->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Simulation is processed in stages with each stage going one after another - eat food/produce food, apply damage, kill organisms, get observations, think decisions, make decisions, try produce children</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The food values are not discrete. Instead, the food is a float value. Each producer can add to this value until it reaches max_food value. If the amount of food is less than food_threshold, then organisms cannot eat this food, and it wi"
                        "ll not be displayed on screen. If organism try to eat more food than there exists in world block, then it will eat only the available food amount.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">There are two modes of organism's movement - discrete and continuous:</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\"\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">With discrete movement organisms move 1 world block in one tick in any of 4 directions. </li>\n"
"<li style=\"\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">With continuous mode the organisms have a velocity. Each tick to the organism's velocity a force is applied (If it has mover) which is calculated by brain (otherwise a random force is applied). T"
                        "he organism will maintain this velocity and move through space until it meets an obstacle. </li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(tab_2), QCoreApplication::translate("Info", "Simulation Info", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Info: public Ui_Info {};
} // namespace Ui

QT_END_NAMESPACE

#endif // INFOWINDOWUI_H
