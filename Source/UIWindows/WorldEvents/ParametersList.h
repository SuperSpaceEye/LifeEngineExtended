//
// Created by spaceeye on 06.08.22.
//

#ifndef LIFEENGINEEXTENDED_PARAMETERSLIST_H
#define LIFEENGINEEXTENDED_PARAMETERSLIST_H

#include <vector>
#include <string>
#include <boost/unordered_map.hpp>

#include "../../Containers/CPU/SimulationParameters.h"
#include "../../Containers/CPU/OrganismBlockParameters.h"
#include "../../Containers/CPU/OrganismInfoContainer.h"
#include "WorldEventsEnums.h"

#include "SimulationParametersTypes.h"

struct ChangeableReturn {
    ValueType type = ValueType::NONE;
    float * float_val = nullptr;
    int   * int_val   = nullptr;
    ClampModes clamp_mode = ClampModes::NoClamp;
    float min_float_clamp = 0;
    float max_float_clamp = 0;
    int   min_int_clamp   = 0;
    int   max_int_clamp   = 0;
};

struct ChangingReturn {
    ValueType type = ValueType::NONE;
    double  * double_val = nullptr;
    int64_t * int64_val  = nullptr;
};

class ParametersList {
private:
    SimulationParameters * sp = nullptr;
    OrganismBlockParameters * bp = nullptr;
    OrganismInfoContainer * ic = nullptr;

    //changeable parameters (simulation parameters, block parameters)
    std::vector<std::string> changeable_pm_list;
    boost::unordered_map<std::string, ChangeableReturn> changeable_pm;
    //changing parameters (simulation parameters, block parameters)
    std::vector<std::string> changing_pm_list;
    boost::unordered_map<std::string, ChangingReturn> changing_pm;

public:
    explicit ParametersList(SimulationParameters * sp, OrganismBlockParameters * bp, OrganismInfoContainer * ic);

    //Returns vector containing a pair of name and type of parameter that can be changed (some simulation parameters).
    const std::vector<std::string> & get_changeable_parameters_list();

    //Returns vector containing a pair of name and type of parameters that are changing in simulation (info).
    const std::vector<std::string> & get_changing_parameters_list();

    ChangeableReturn get_changeable_value_address_from_name(std::string & name);
    ChangingReturn get_changing_value_address_from_name(std::string & name);


};

#endif //LIFEENGINEEXTENDED_PARAMETERSLIST_H
