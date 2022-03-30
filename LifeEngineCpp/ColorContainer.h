//
// Created by spaceeye on 20.03.2022.
//

#ifndef THELIFEENGINECPP_COLORCONTAINER_H
#define THELIFEENGINECPP_COLORCONTAINER_H

#include "SFML/Graphics.hpp"

struct ColorContainer {
    sf::Color menu_color = sf::Color{200, 200, 255};
    sf::Color simulation_background_color = sf::Color(70, 70, 150);

    sf::Color empty_block = sf::Color{0,  0,  0};
    sf::Color mouth =       sf::Color{255,188,0};
    sf::Color producer =    sf::Color{0,  200,0};
    sf::Color mover =       sf::Color{0,  235,211};
    sf::Color killer =      sf::Color{255,60, 112};
    sf::Color armor =       sf::Color{125,38, 255};
    sf::Color eye =         sf::Color{210,180,255};
    sf::Color food =        sf::Color{125,186,255};
    sf::Color wall =        sf::Color{70, 70, 70};
};


#endif //THELIFEENGINECPP_COLORCONTAINER_H
