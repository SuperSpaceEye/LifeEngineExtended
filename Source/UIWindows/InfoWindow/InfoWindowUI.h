/********************************************************************************
** Form generated from reading UI file 'info.ui'
**
** Created by: Qt User Interface Compiler version 6.4.0
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
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"LifeEngine_Extended_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">L</span><span style=\" font-size:xx-large; font-weight:600;\">ifeEngine Extended</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">C++ implementation of <a href=\"https://github.com/MaxRobinsonTheGreat/LifeEngine\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/MaxRobinsonTheGreat/LifeEngine</span></a> that extends the original version with new feature"
                        "s.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This version is work in progress.</p>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Important_information_5\"></a><span style=\" font-size:xx-large; font-weight:600;\">I</span><span style=\" font-size:xx-large; font-weight:600;\">mportant information:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You should press enter after inputing text in line edits or it will not register.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Setting -1 into max fps/sps/organisms line edits will disable the limit.</li>\n"
"<li s"
                        "tyle=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">To use keyboard actions you should click on simulation window first.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When you use \342\200\234Choose organism\342\200\235 mouse mode it will search for any organism_index in brush area and will stop when it finds one.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can use mouse actions in editor.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Keyboard actions do not work in editor.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Mouse_actions_13\"></a><span style=\" font-size:xx-large; font-weight:6"
                        "00;\">M</span><span style=\" font-size:xx-large; font-weight:600;\">ouse actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold left button to place/kill.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold right button to delete/kill</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold scroll button to move the view.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Scroll to zoom in/out.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Keyboard_button_actions_19\">"
                        "</a><span style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">eyboard button actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234W\342\200\235 or \342\200\234Up\342\200\235 - move view up.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234A\342\200\235 or \342\200\234Left\342\200\235 - move view left.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234S\342\200\235 or \342\200\234Down\342\200\235 - move view down.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234D\342"
                        "\200\235 or \342\200\234Right\342\200\235 - move view right.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Shift\342\200\235 - hold to make movement using keys 2 times as fast.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234R\342\200\235 - reset view.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234M\342\200\235 - hide/show menu.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Space bar\342\200\235 - pause simulation,</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234.\342\200\235 - pass one simulation tick.</li>\n"
"<li style=\" margin-top:0p"
                        "x; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234numpad +\342\200\235 - zoom in.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234numpad -\342\200\235 - zoom out.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2340\342\200\235 - switch mouse mode to no action.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2341\342\200\235 - switch mouse mode to place/delete food.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2342\342\200\235 - switch mouse mode to kill organisms.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent"
                        ":0px;\">\342\200\2343\342\200\235 - switch mouse mode to place/delete walls.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2344\342\200\235 - switch mouse mode to place editor organisms.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2345\342\200\235 - switch mouse mode to choose organisms from simulation.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234f11\342\200\235 - enable/disable fullscreen.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Known_bugs_37\"></a><span style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">nown bugs:</span></h1>\n"
"<ul style=\"margin-top: 0px; "
                        "margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Saving and loading will not work correctly unless your path contains only english letters.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Renderer imprecisely calculates textures for small num of pixels per block. So if you want to have nice, correct images/videos when using recorder, set &quot;Number of pixels per world block&quot; &gt; 10.</li></ul>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Saving and loading will not work correctly unless your path contains only english letters.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; "
                        "margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The program will crash sometimes.</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(about_tab), QCoreApplication::translate("Info", "About", nullptr));
        textEdit_2->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Evolution_Controls_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">E</span><span style=\" font-size:xx-large; font-weight:600;\">volution Controls</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">These are the explanations for some evolution options</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt"
                        "-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Lifespan multiplier\342\200\235</span> - Multiplicator of the sum of \342\200\234Lifetime Weight\342\200\235 of each block of an organism.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Anatomy mutation rate mutation step\342\200\235</span> - An amount by which a mutation rate of an organism can increase or decrease</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Anatomy mutation rate delimiter\342\200\235</span> - A parameter which controls whatever organism\342\200\231s anatomy mutation rate will be biased to increase or decrease. If &gt;0.5 then the rate will increase, if &lt;0.5 then the rate will decrease, if == 0.5 then no bias.</li>\n"
"<li style=\" margin-top:0px; mar"
                        "gin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Brain mutation rate mutation step\342\200\235</span> - The same as \342\200\234Anatomy mutation rate mutation step\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Brain mutation rate delimiter\342\200\235</span> - The same as \342\200\234Anatomy mutation rate delimiter\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Fix reproduction distance\342\200\235</span> - Will make reproduction distance always equal to \342\200\234Min reproduction distance\342\200\235 during reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent"
                        ":0px;\"><span style=\" font-weight:600;\">\342\200\234Organism\342\200\231s self cells block sight\342\200\235</span> - If disabled, organism can see through itself. If enabled, the eye that points to the cell that belongs to itself will return \342\200\234Empty block\342\200\235 as observation.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Set fixed move range\342\200\235</span> - Will force organisms to use \342\200\234Min move range\342\200\235 and will make child move ranges equal to parent during reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Move range delimiter\342\200\235</span> - The same as Anatomy mutation rate delimiter\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-bl"
                        "ock-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Failed reproduction eats food\342\200\235</span> - If disabled, then the food will be deduced from parent organism only after successful reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Rotate every move tick\342\200\235</span> - Will make organisms rotate every time they move. If disabled, then they will rotate only at the end of move range.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Simplified food production\342\200\235</span> - Will try to produce food for each space that can produce food.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200"
                        "\234Eat first, then produce food\342\200\235</span> - If disabled, will produce food first and only then eat.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Use new child position calculator\342\200\235</span> - New child position calculator calculating position of a child by first calculating the coordinates of edges of parent and child + distance. For example, if the chosen reproduction direction is up, then calculator will calculate the uppermost y cell coordinate of a parent, the bottom y cell coordinate of a child + distance. That way, the child organism will never appear inside a parent organism_index. The old child position calculator however calculates only the edge coordinates of a parent organism_index + distance, allowing child organisms to appear inside parent, with the side effect of organisms being unable to reproduce if the reproducing distance is less than (height or"
                        " width)/2 (depending on child organism_index rotation and chosen reproductive direction)</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Check if path is clear\342\200\235</span> - If enabled, will check for each cell of a child organism if there is an obstruction in the way (except for parents cells), like a wall (If &quot;food blocks reproduction&quot; is enabled, will only check for existance of food at the end of the path). If there is, then the parent organism_index will not reproduce. If disabled, the child will just check if there is a space for itself at the end coordinated, but it will introduce some behaviours such as child organisms hopping though walls if they are thin enough.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Recenter organisms t"
                        "o imaginary position\342\200\235</span> - If disabled, will set the center of an organism to the existing block, which is the closest to the 0,0 position. If enabled, will set the center of an organism to x=organism_width/2, y=organism_height/2 even if there is no block at that position.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Organism cell parameters modifiers\342\200\235 - modifiers for each cell of all organisms.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Life point\342\200\235</span> - The amount of life points this cell will give to organism</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-inde"
                        "nt:0px;\"><span style=\" font-weight:600;\">\342\200\234Lifetime weight\342\200\235</span> - The amount of lifetime this cell will give to organism</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Chance Weight\342\200\235</span> - Controls how likely this cell will be picked during reproduction compared to others. If 0, the cell will never get picked.</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(evolution_controls_tab), QCoreApplication::translate("Info", "Evolution Controls", nullptr));
        textEdit_3->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Settings_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">S</span><span style=\" font-size:xx-large; font-weight:600;\">ettings</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Float number precision\342\200\235</span> - A decorative parameter that control how many zeros after the decimal point of f"
                        "loats will be displayed in labels</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Wait for engine to stop to render\342\200\235</span> - If enabled, will send an engine a signal to stop before rendering simulation grid.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Really stop render\342\200\235</span>- To render an image, the program first calculates what cells will be seen by user, and then it copies them to the secondary grid containing only cell type and rotation, from which the image is constructed. If disabled, will parse the whole grid when \342\200\234Stop render\342\200\235 button is pressed, which will allow to move and scale the view. If enabled, will not parse the grid or construct an image when \342\200\234Stop render\342\200\235 is pressed."
                        "</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(settings_tab), QCoreApplication::translate("Info", "Settings", nullptr));
        textEdit_4->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Recorder_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">R</span><span style=\" font-size:xx-large; font-weight:600;\">ecorder</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Recorder can either create photos or videos of full grid. To create a photo just click on \342\200\234Create image\342\200\235.<br />To make a video follow these steps:</p>\n"
"<ol style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-inden"
                        "t: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">\342\200\234New recording\342\200\235</span></li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">\342\200\234Start recording\342\200\235</span></li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Record for however long you want.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">\342\200\234Pause recording\342\200\235</span></li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click <span style=\" font-weight:600;\">\342\200\234Compile intermediate data"
                        " into video\342\200\235</span></li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Wait until video is compiled. Finished video will be in the \342\200\234path to program dir/videos\342\200\235. The video will be named with the timestamp of the start of the recording</li></ol>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Warning! The recorder right now is pretty inefficient. During the recording it will quickly fill up a lot of space, while the creating of video is pretty slow, so it is preferable to keep your recording &lt;5000 simulation ticks.<br />You can compile several videos in parallel, but it is not recommended.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Various_setting_13\"></a><span style=\" font-size:large; font-weight:600;\">V</span><span st"
                        "yle=\" font-size:large; font-weight:600;\">arious setting.</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Grid tbuffer size\342\200\235</span> - The size of a tbuffer. The program will first write grid states to the tbuffer, before writing all at once to the drive.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Log every n tick\342\200\235</span> - Will log every n tick.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Video output FPS\342\200\235</span> - FPS that will be set during video construction.</li></ul>\n"
""
                        "<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Various_buttons_18\"></a><span style=\" font-size:large; font-weight:600;\">V</span><span style=\" font-size:large; font-weight:600;\">arious buttons.</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234New recording\342\200\235</span> - Will create new folder in /temp/ with timestamp where recording will be stored.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Stop recording\342\200\235</span> - Will stop the recording, flushing data in the tbuffer to the disk.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:"
                        "0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Clear intermediate data\342\200\235</span> - Will stop the recording before freeing the tbuffer space.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Delete all intermediate data from disk\342\200\235</span> - Will delete everything in the /temp/ folder.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Compile intermediate data into video\342\200\235</span> - The output will be in /videos/ folder. Compilation is done in two stages:</li>\n"
"<ol style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 2;\"><li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-bl"
                        "ock-indent:0; text-indent:0px;\">Convert recording data into a series of images in /temp/ folder. This stage is the slowest, though you can stop the compilation and continue later.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Convert series of images into video. This stage is magnitudes faster than the previous, but cannot be stopped without losing the progress.</li></ol>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Load intermediate data location\342\200\235</span> - Choose the folder with the recording.</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(recorder_tab), QCoreApplication::translate("Info", "Recorder", nullptr));
        textEdit_5->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"World_Events_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">W</span><span style=\" font-size:xx-large; font-weight:600;\">orld Events</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events are events that are executed after simulation tick.<br />Right now there is two types of events: \342\200\234Conditional\342\200\235, \342\200\234Change Value\342\200\235.<br />World events are divided into rows and nodes. First executed rows from the bott"
                        "om, then inside rows  nodes are executed left to right.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events have two settings - \342\200\234Update World Events every n ticks\342\200\235 and \342\200\234Collect Info every n ticks\342\200\235.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Update World Events every n ticks\342\200\235</span> - Although world events are pretty lightweight in regard to performance, they are not free. So I made this parameter to control when nodes are updated.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Collect Info every n ti"
                        "cks\342\200\235</span> - The conditional node needs info to make decision.<br />If the value of this parameter is too large the conditional node will use an outdated data.<br />However, if the value is too small it will hurt the performance, as gathering data of simulation is not free.<br />Be aware, that because world events and statistics share the same location of info, when UI part sends signal to update data, it will also update data for the world events.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Every node has \342\200\234Update every n tick\342\200\235. It works the same as \342\200\234Update World Events every n ticks\342\200\235. When last execution time exceeds this parameter, the node will execute.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Each world events branch also has \342\200\234Repeat branch\342\200\235. If even"
                        "t branch reaches the end and parameter was toggled, the execution will begin from the start, else it will just stop.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">World events in \342\200\234World Events Editor\342\200\235 and simulation are separate, so changes in the editor will not affect world events in simulation, unless you apply them.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Conditional_Event_18\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">onditional Event</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When event branch reaches conditional node, it will continuously check if the statement is true. If it is, the execution of the next node will begin, otherwise i"
                        "t will repeat the check.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Change_Value_21\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">hange Value</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Change value node allows for World Events to actually influence the simulation.<br />With this node you can change \342\200\234some\342\200\235 simulation and block parameters.<br />This node has several modes with how it can change selected value.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Linear\342\200\235</span> - Will change t"
                        "he value to target across time.<br />The parameter \342\200\234Time horizon\342\200\235 controls for how long the value is changed.<br />During the execution of this mode, any changes to the value will not be applied.<br />If two Linear nodes started executing at the same time, and have the same time horizon, the one in higher branch will set final target value, otherwise the one finishing last will set the value.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Step\342\200\235</span> - Will change the value to target value upon reaching.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The modes below were added by omgdev. All nodes are executed upon reaching.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-t"
                        "op:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Increase by\342\200\235</span> - Will increase chosen variable by target amount.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Decrease by\342\200\235</span> - Will decrease chosen variable by target amount.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Multiply by\342\200\235</span> - Will multiply chosen variable by target amount.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Divide by\342\200\235</span> - Will divide chosen variable by target amount.</li></ul>\n"
"<h3 style=\""
                        " margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Running_World_Events_37\"></a><span style=\" font-size:large; font-weight:600;\">R</span><span style=\" font-size:large; font-weight:600;\">unning World Events</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">After creating World Events click \342\200\234Apply events\342\200\235.<br />World events will not be applied if there are nodes without set value.<br />If applying was successful, the world events will start execution.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">While the world events are running, you can\342\200\231t change some values.<br />To change the values, pause the simulation or pause execution of world events in the tab \342\200\234Current World Events Viewer\342\200\235 of world events window.</p>\n"
""
                        "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">If simulation resets, world events will also automatically reset and start from the beginning. If execution of world events is stopped or the simulation resets, the simulation settings will be set to the state they were before the execution of world events started, unless you stop them with \342\200\234Stop Events No Setting Reset\342\200\235.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can pause/resume execution of world events with buttons \342\200\234Pause events\342\200\235/\342\200\234Resume events\342\200\235. These buttons will not reset world events.<br />If world events are already applied and were stopped, you can use \342\200\234Start Events\342\200\235 button to re-enable already applied world events.</p></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(world_events_tab), QCoreApplication::translate("Info", "World Events", nullptr));
        textEdit_7->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Organism_Construction_Code_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">O</span><span style=\" font-size:xx-large; font-weight:600;\">rganism Construction Code</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Organism Construction Code or OCC is a way to represent anatomy of organisms as a DNA like structure.<br />The OCC is an array of different instructions, executed one after another. The anatomy is constructed on a grid, with \342\200\234curs"
                        "or\342\200\235 being the main component. During the process of anatomy constructions, different elements are used:</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Construction_elements_3\"></a><span style=\" font-size:large; font-weight:600;\">C</span><span style=\" font-size:large; font-weight:600;\">onstruction elements:</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234cursor\342\200\235 - a point in space depending on a position of which organism blocks are placed.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234origin\342\200\235 - a changeable position that a cursor can return to if appropriate command is executed"
                        ".</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234rotation cursor\342\200\235 - the base rotation of blocks that will be placed with.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234group\342\200\235 - a sequence of random instruction with the size randomly chosen between [1, max_size].</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The OCC can be viewed and edited in an \342\200\234edit OCC\342\200\235 option in editor window.</p>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Rules_of_OCC_edit_window_10\"></a><span style=\" font-size:large; font-weight:600;\">R</span><span style=\" font-size:large; font-weight:600;\">ules of OCC edit window:</"
                        "span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">To write a sequence of instructions, each instruction must be a valid instruction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Instruction must be finished with a \342\200\234;\342\200\235 sign.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The spaces, indentation or new lines do not affect the execution of OCC.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can make comment with a \342\200\234/\342\200\235 sign.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-"
                        "indent:0; text-indent:0px;\">After execution, OCC must create at least 1 organism block.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Instructions can be written in either full or shortened form.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"All_OCC_mutations_18\"></a><span style=\" font-size:large; font-weight:600;\">A</span><span style=\" font-size:large; font-weight:600;\">ll OCC mutations:</span></h3>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Append random - appends a group to the end of OCC sequence.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">I"
                        "nsert random - inserts a group to a random uniformly sampled position in OCC sequence.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Change random - first gets a group, then randomly chooses the position inside the OCC sequence, then overwrites the part of existing OCC with instructions from group until either a group is fully written, or OCC reaches end.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Delete random - will delete OCC instruction the size of group starting from random position, until either the number of instructions deleted = group size, or OCC reaches end.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Swap random - will choose random position, randomly decide the direction of movement (left or right), randomly decide the distance instructions "
                        "will be moved, randomly decide the distance from chosen position, then it will start to swap elements of group size, until either it successfully swapped the elements, or OCC ends.</li></ul>\n"
"<h3 style=\" margin-top:14px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"All_OCC_instructions_25\"></a><span style=\" font-size:large; font-weight:600;\">A</span><span style=\" font-size:large; font-weight:600;\">ll OCC instructions:</span></h3>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Shift Instructions - Shifts cursor to direction, or if there is SetBlock instruction after it, sets the block above the cursor, not changing it\342\200\231s position.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-inde"
                        "nt:0px;\">ShiftUp or SU</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftUpLeft or SUL</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftLeft or SL</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftLeftDown or SLD</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftDown or SD</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftDownRight or SDR</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ShiftRight or SR</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-i"
                        "ndent:0; text-indent:0px;\">ShiftRightUp or SRU</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Apply rotation instructions - Sets the rotation of rotation cursor to the new rotation.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationUp or ARU</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationLeft or ARL</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationDown or ARD</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ApplyRotationRight or ARR</li></ul>\n"
"<p sty"
                        "le=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set rotation instructions - Sets the rotation of a block directly underneath the cursor to the new rotation. Will work only if there is already a block underneath the cursor.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationUp or SRTU</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationLeft or SRL</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationDown or SRD</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetRotationRight or SRR</li></ul"
                        ">\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">ResetToOrigin or RTO -  Resets the position of a cursor to the position of origin.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetOrigin or SO -  Sets the position of an origin to the position of a cursor.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set block instructions - -  Sets the block on a grid to type.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockMouth or SBM</li>\n"
"<li style=\" margin-top:0p"
                        "x; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockProducer or SBP</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockMover or SBMV</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockKiller or SBK</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockArmor or SBA</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetBlockEye or SBE</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Set under block instructions -  Sets the block on a grid to mouth directly underneath of the cursor. Is not affected by shift instructions</p>\n"
"<ul style=\""
                        "margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockMouth or SUBM</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockProducer or SUBP</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockMover or SUBMV</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockKiller or SURK</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBlockArmor or SUBA</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SetUnderBl"
                        "ockEye or SUBE</li></ul></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(tab), QCoreApplication::translate("Info", "OCC", nullptr));
        textEdit_6->setHtml(QCoreApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Custom_Textures_0\"></a><span style=\" font-size:xx-large; font-weight:600;\">C</span><span style=\" font-size:xx-large; font-weight:600;\">ustom Textures</span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Life Engine Extended allows you to set arbitrary images as textures for organism cells, food, walls and empty space.<br />For image to be usable by program, it needs to be renamed to one of these titles: <span style=\" font-weight:600;\">(\342\200\234empty\342\200"
                        "\235, \342\200\234mouth\342\200\235, \342\200\234producer\342\200\235, \342\200\234mover\342\200\235, \342\200\234killer\342\200\235, \342\200\234armor\342\200\235, \342\200\234eye\342\200\235, \342\200\234food\342\200\235, \342\200\234wall\342\200\235)</span> and be placed into <span style=\" font-weight:600;\">\342\200\234/textures/\342\200\235</span> folder.<br />These textures will be loaded upon program loading, or if you click <span style=\" font-weight:600;\">\342\200\234Update textures\342\200\235</span>.<br />If there is no image with type in <span style=\" font-weight:600;\">\342\200\234/textures/\342\200\235</span> folder, the program will revert to base texture.<br />The images can be of any size, but the program will stretch them to box shape.</p></body></html>", nullptr));
        OCC->setTabText(OCC->indexOf(custom_textures_tab), QCoreApplication::translate("Info", "Custom Textures", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Info: public Ui_Info {};
} // namespace Ui

QT_END_NAMESPACE

#endif // INFOWINDOWUI_H
