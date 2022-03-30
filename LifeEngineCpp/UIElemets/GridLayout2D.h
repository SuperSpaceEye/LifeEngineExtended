//
// Created by spaceeye on 26.03.2022.
//

#ifndef THELIFEENGINECPP_GRIDLAYOUT2D_H
#define THELIFEENGINECPP_GRIDLAYOUT2D_H

#include "TGUI/TGUI.hpp"
#include "vector"

struct Container {
    tgui::Widget::Ptr widget;
    uint x;
    uint y;
};

class GridLayout2D {
private:
    static std::vector<double> linspace(double start, double end, int num);

    std::vector<double> width_values;
    std::vector<double> height_values;

    uint num_rows;
    uint num_cols;
    uint left_pad_x;
    uint right_pad_x;
    uint up_pad_y;
    uint down_pad_y;
    tgui::Widget::Ptr base_widget;
    tgui::GuiSFML &gui;
    std::vector<Container> widgets;

public:
    GridLayout2D(unsigned int num_row, unsigned int num_col, tgui::Widget::Ptr base_widget, tgui::GuiSFML &gui,
                 uint left_pad_x=0, uint right_pad_x=0, uint up_pad_y=0, uint down_pad_y=0);
    void add(tgui::Widget::Ptr widget_to_tie, uint x, uint y);

    void update_positions();
    void setVisible(bool visible);
    void setEnabled(bool enabled);
};


#endif //THELIFEENGINECPP_GRIDLAYOUT2D_H
