//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include <QColor>

struct ColorContainer {
    QColor menu_color = QColor{200, 200, 255};
    QColor simulation_background_color = QColor(70, 70, 150);

    QColor empty_block = QColor{0,  0,  0};
    QColor mouth =       QColor{255,188,0};
    QColor producer =    QColor{0,  200,0};
    QColor mover =       QColor{0,  235,211};
    QColor killer =      QColor{255,60, 112};
    QColor armor =       QColor{125,38, 255};
    QColor eye =         QColor{210,180,255};
    QColor food =        QColor{125,186,255};
    QColor wall =        QColor{70, 70, 70};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
