/********************************************************************************
** Form generated from reading UI file 'info.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
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
    QTabWidget *tabWidget;
    QWidget *about_tab;
    QHBoxLayout *horizontalLayout;
    QTextEdit *textEdit;
    QWidget *evolution_controls_tab;
    QVBoxLayout *verticalLayout_2;
    QTextEdit *textEdit_2;
    QWidget *settings_tab;
    QVBoxLayout *verticalLayout_3;
    QTextEdit *textEdit_3;

    void setupUi(QWidget *Info)
    {
        if (Info->objectName().isEmpty())
            Info->setObjectName(QString::fromUtf8("Info"));
        Info->resize(855, 739);
        verticalLayout = new QVBoxLayout(Info);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(Info);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        about_tab = new QWidget();
        about_tab->setObjectName(QString::fromUtf8("about_tab"));
        horizontalLayout = new QHBoxLayout(about_tab);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        textEdit = new QTextEdit(about_tab);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));

        horizontalLayout->addWidget(textEdit);

        tabWidget->addTab(about_tab, QString());
        evolution_controls_tab = new QWidget();
        evolution_controls_tab->setObjectName(QString::fromUtf8("evolution_controls_tab"));
        verticalLayout_2 = new QVBoxLayout(evolution_controls_tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        textEdit_2 = new QTextEdit(evolution_controls_tab);
        textEdit_2->setObjectName(QString::fromUtf8("textEdit_2"));

        verticalLayout_2->addWidget(textEdit_2);

        tabWidget->addTab(evolution_controls_tab, QString());
        settings_tab = new QWidget();
        settings_tab->setObjectName(QString::fromUtf8("settings_tab"));
        verticalLayout_3 = new QVBoxLayout(settings_tab);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        textEdit_3 = new QTextEdit(settings_tab);
        textEdit_3->setObjectName(QString::fromUtf8("textEdit_3"));

        verticalLayout_3->addWidget(textEdit_3);

        tabWidget->addTab(settings_tab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Info);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Info);
    } // setupUi

    void retranslateUi(QWidget *Info)
    {
        Info->setWindowTitle(QApplication::translate("Info", "Info Window", nullptr));
        textEdit->setHtml(QApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
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
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When you use \342\200\234Choose organism\342\200\235 mouse mode it will search for any organism in brush area and will stop when it finds one.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You can use mouse actions in editor.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Keyboard actions do not work in editor.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Mouse_actions_13\"></a><span style=\" font-size:xx-large; font-weight:600;\">"
                        "M</span><span style=\" font-size:xx-large; font-weight:600;\">ouse actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold left button to place/kill.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold right button to delete/kill</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hold scroll button to move the view.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Scroll to zoom in/out.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Keyboard_button_actions_19\"></a><s"
                        "pan style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">eyboard button actions:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234W\342\200\235 or \342\200\234Up\342\200\235 - move view up.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234A\342\200\235 or \342\200\234Left\342\200\235 - move view left.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234S\342\200\235 or \342\200\234Down\342\200\235 - move view down.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234D\342\200\235"
                        " or \342\200\234Right\342\200\235 - move view right.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Shift\342\200\235 - hold to make movement using keys 2 times as fast.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234R\342\200\235 - reset view.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234M\342\200\235 - hide/show menu.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Space bar\342\200\235 - pause simulation,</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234.\342\200\235 - pass one simulation tick.</li>\n"
"<li style=\" margin-top:0px; margi"
                        "n-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234numpad +\342\200\235 - zoom in.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234numpad -\342\200\235 - zoom out.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2341\342\200\235 - switch mouse mode to place/delete food.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2342\342\200\235 - switch mouse mode to kill organisms.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2343\342\200\235 - switch mouse mode to place/delete walls.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-inden"
                        "t:0px;\">\342\200\2344\342\200\235 - switch mouse mode to place editor organisms.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\2345\342\200\235 - switch mouse mode to choose organisms from simulation.</li></ul>\n"
"<h1 style=\" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a name=\"Known_bugs_37\"></a><span style=\" font-size:xx-large; font-weight:600;\">K</span><span style=\" font-size:xx-large; font-weight:600;\">nown bugs:</span></h1>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Saving and loading will not work correctly unless your path contains only english letters.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -q"
                        "t-block-indent:0; text-indent:0px;\">Mouse movement tracking is imprecise.</li></ul></body></html>", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(about_tab), QApplication::translate("Info", "About", nullptr));
        textEdit_2->setHtml(QApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">These are the explanations for some evolution options</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Lifespan multiplier\342\200\235</span> - Multiplicator of the sum of \342\200\234Lifetime Weight\342\200\235 of each block of an organism.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px"
                        "; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Anatomy mutation rate mutation step\342\200\235</span> - An amount by which a mutation rate of an organism can increase or decrease</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Anatomy mutation rate delimiter\342\200\235</span> - A parameter which controls whatever organism\342\200\231s anatomy mutation rate will be biased to increase or decrease. If &gt;0.5 then the rate will increase, if &lt;0.5 then the rate will decrease, if == 0.5 then no bias.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Brain mutation rate mutation step\342\200\235</span> - The same as \342\200\234Anatomy mutation rate mutation step\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bot"
                        "tom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Brain mutation rate delimiter\342\200\235</span> - The same as \342\200\234Anatomy mutation rate delimiter\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Fix reproduction distance\342\200\235</span> - Will make reproduction distance always equal to \342\200\234Min reproduction distance\342\200\235 during reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Organism\342\200\231s self cells block sight\342\200\235</span> - If disabled, organism can see through itself. If enabled, the eye that points to the cell that belongs to itself will return \342\200\234Empty block\342\200\235 as observation.</li>\n"
"<li s"
                        "tyle=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Set fixed move range\342\200\235</span> - Will force organisms to use \342\200\234Min move range\342\200\235 and will make child move ranges equal to parent during reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Move range delimiter\342\200\235</span> - The same as Anatomy mutation rate delimiter\342\200\235</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Failed reproduction eats food\342\200\235</span> - If disabled, then the food will be deduced from parent organism only after successful reproduction.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-"
                        "right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Rotate every move tick\342\200\235</span> - Will make organisms rotate every time they move. If disabled, then they will rotate only at the end of move range.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Simplified food production\342\200\235</span> - Will try to produce food for each space that can produce food.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Eat first, then produce food\342\200\235</span> - If disabled, will produce food first and only then eat.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Use new child position"
                        " calculator\342\200\235</span> - New child position calculator calculating position of a child by first calculating the coordinates of edges of parent and child + distance. For example, if the chosen reproduction direction is up, then calculator will calculate the uppermost y cell coordinate of a parent, the bottom y cell coordinate of a child + distance. That way, the child organism will never appear inside a parent organism. The old child position calculator however calculates only the edge coordinates of a parent organism + distance, allowing child organisms to appear inside parent, with the side effect of organisms being unable to reproduce if the reproducing distance is less than (height or width)/2 (depending on child organism rotation and chosen reproductive direction)</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Check if path is clear\342\200\235</span> - If enabled, will"
                        " check for each cell of a child organism if there is an obstruction in the way (except for parents cells), like a wall or a food if \342\200\234Food blocks reproduction\342\200\235 is enabled. If there is, then the parent organism will not reproduce. If disabled, the child will just check if there is a space for itself at the end coordinated, but it will introduce some behaviours such as child organisms hopping though walls if they are thin enough.</li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\342\200\234Organism cell parameters modifiers\342\200\235 - modifiers for each cell of all organisms.</p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Life point\342\200\235</span> - The amount of l"
                        "ife points this cell will give to organism</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Lifetime weight\342\200\235</span> - The amount of lifetime this cell will give to organism</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Chance Weight\342\200\235</span> - Controls how likely this cell will be picked during reproduction compared to others. If 0, the cell will never get picked.</li></ul></body></html>", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(evolution_controls_tab), QApplication::translate("Info", "Evolution Controls", nullptr));
        textEdit_3->setHtml(QApplication::translate("Info", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Float number precision\342\200\235</span> - A decorative parameter that control how many zeros after the decimal point of floats will be displayed in labels</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Wait for engine to stop to render\342\200\235</span> - If enabled, will send an "
                        "engine a signal to stop before rendering simulation grid.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Simplified rendering\342\200\235</span> - If enabled, will not render eyes. Will be removed soon.</li>\n"
"<li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">\342\200\234Really stop render\342\200\235</span>- To render an image, the program first calculates what cells will be seen by user, and then it copies them to the secondary grid containing only cell type and rotation, from which the image is constructed. If disabled, will parse the whole grid when \342\200\234Stop render\342\200\235 button is pressed, which will allow to move and scale the view. If enabled, will not parse the grid or construct an image when \342\200\234Stop render\342\200\235 is pressed.</li></ul></body"
                        "></html>", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(settings_tab), QApplication::translate("Info", "Settings", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Info: public Ui_Info {};
} // namespace Ui

QT_END_NAMESPACE

#endif // INFOWINDOWUI_H
