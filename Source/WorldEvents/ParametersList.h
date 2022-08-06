//
// Created by spaceeye on 06.08.22.
//

#ifndef LIFEENGINEEXTENDED_PARAMETERSLIST_H
#define LIFEENGINEEXTENDED_PARAMETERSLIST_H

#include <vector>
#include <string>
#include <boost/unordered_map.hpp>

#include "../Containers/CPU/SimulationParameters.h"
#include "../Containers/CPU/OrganismBlockParameters.h"

#include "SimulationParametersTypes.h"

struct ReturnValue {
    ValueType type = ValueType::NONE;
    float * float_val = nullptr;
    int   * int_val   = nullptr;
};

class ParametersList {
private:
    SimulationParameters * sp = nullptr;
    OrganismBlockParameters * bp = nullptr;

    //changeable parameters (simulation parameters, block parameters)
    boost::unordered_map<std::string, ReturnValue> changeable_pm;
    //changing parameters (simulation parameters, block parameters)
    boost::unordered_map<std::string, ReturnValue> changing_pm;

public:
    explicit ParametersList(SimulationParameters * sp, OrganismBlockParameters * bp);

    //Returns vector containing a pair of name and type of parameter that can be changed (some simulation parameters).
    std::vector<std::pair<std::string, std::string>> get_changeable_parameters_list();

    //Returns vector containing a pair of name and type of parameters that are changing in simulation (info).
    std::vector<std::pair<std::string, std::string>> get_changing_parameters_list();

    ReturnValue get_changeable_value_address_from_name(std::string & name);
    ReturnValue get_changing_value_address_from_name(std::string & name);


};

#endif //LIFEENGINEEXTENDED_PARAMETERSLIST_H
