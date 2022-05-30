//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include <QColor>

struct ColorContainer {
    //QColor menu_color = QColor{200, 200, 255};
    QColor simulation_background_color = QColor(58, 75, 104);
    QColor organism_boundary = QColor{255, 0, 0};

    QColor empty_block = QColor{14, 19, 24};
    QColor mouth =       QColor{222,177,77};
    QColor producer =    QColor{21, 222,89};
    QColor mover =       QColor{96, 212,255};
    QColor killer =      QColor{248,35, 128};
    QColor armor =       QColor{114,48, 219};
    QColor eye =         QColor{182,193,234};
    QColor food =        QColor{47, 122,183};
    QColor wall =        QColor{128, 128, 128};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
