//
// Created by spaceeye on 26.03.2022.
//

#include "GridLayout2D.h"

#include <utility>

GridLayout2D::GridLayout2D(unsigned int num_row, unsigned int num_col, tgui::Widget::Ptr base_widget,
                           tgui::GuiSFML &gui, uint left_pad_x, uint right_pad_x, uint up_pad_y, uint down_pad_y):
                           num_rows(num_row), num_cols(num_col), base_widget(base_widget), gui(gui), left_pad_x(left_pad_x),
                           right_pad_x(right_pad_x), up_pad_y(up_pad_y), down_pad_y(down_pad_y){
    widgets = std::vector<Container>{};
}

std::vector<double> GridLayout2D::linspace(double start, double end, int num) {
    std::vector<double> linspaced;

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}

void GridLayout2D::add(tgui::Widget::Ptr widget_to_tie, uint x, uint y) {
    if (x >= num_rows || y >= num_cols) {return;}

    auto width = linspace(base_widget->getPosition().x+left_pad_x,
                          base_widget->getPosition().x+base_widget->getSize().x-widget_to_tie->getSize().x-right_pad_x,
                          num_rows);
    auto height= linspace(base_widget->getPosition().y+up_pad_y,
                          base_widget->getPosition().y+base_widget->getSize().y-widget_to_tie->getSize().y-down_pad_y,
                          num_cols);

    widget_to_tie->setPosition(width[x], height[y]);
    gui.add(widget_to_tie);
    widgets.push_back(Container{widget_to_tie, x, y});
}

void GridLayout2D::update_positions() {
    //ok that's... ugly.
    for (auto & container: widgets) {
        container.widget->setPosition(
                linspace(
                        base_widget->getPosition().x + left_pad_x,
                        base_widget->getPosition().x + base_widget->getSize().x - container.widget->getSize().x -
                        right_pad_x,
                        num_rows)[container.x],
                linspace(
                        base_widget->getPosition().y + up_pad_y,
                        base_widget->getPosition().y + base_widget->getSize().y - container.widget->getSize().y -
                        down_pad_y,
                        num_cols)[container.y]
        );
    }
}

void GridLayout2D::setVisible(bool visible) {
    for (auto & container: widgets) {
        container.widget->setVisible(visible);
    }
}

void GridLayout2D::setEnabled(bool enabled) {
    for (auto & container: widgets) {
        container.widget->setEnabled(enabled);
    }
}


